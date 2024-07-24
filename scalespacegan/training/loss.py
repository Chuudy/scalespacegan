# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

# added imports
from metrics import equivariance
from util import losses, util, patch_util
import random

# added imports for multiscale consistency loss
from util.misc import isotropic_scaling_matrix_2D_torch

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

def apply_affine_batch(img, transform):
    # hacky .. apply affine transformation with cuda kernel in batch form
    crops = []
    masks = []
    for i, t in zip(img, transform):
        crop, mask = equivariance.apply_affine_transformation(
            i[None], t.inverse())
        crops.append(crop)
        masks.append(mask)
    crops = torch.cat(crops, dim=0)
    masks = torch.cat(masks, dim=0)
    return crops, masks

def apply_affine_batch_and_fix_mask(img, transform, mode='replicate'):
    # hacky .. apply affine transformation with cuda kernel in batch form
    crops = []
    masks = []
    for i, t in zip(img, transform):
        crop, mask = equivariance.apply_affine_transformation(
            i[None], t.inverse())
        mask = fix_mask_with_scale_matrix(mask, t)
        crops.append(crop)
        masks.append(mask)
    crops = torch.cat(crops, dim=0)
    masks = torch.cat(masks, dim=0)
    return crops, masks

def apply_simplified_affine_batch_and_fix_mask(img, transform, mode='constant'):
    # hacky .. apply affine transformation with cuda kernel in batch form
    crops = []
    masks = []
    for i, t in zip(img, transform):
        crop, mask = equivariance.apply_simplified_affine_transformation(
            i[None], t.inverse(), mode=mode)
        mask = fix_mask_with_scale_matrix(mask, t)
        crops.append(crop)
        masks.append(mask)
    crops = torch.cat(crops, dim=0)
    masks = torch.cat(masks, dim=0)
    return crops, masks

def fix_mask(mask, scale):
    mask = torch.zeros_like(mask)
    mask_width = mask.shape[2]
    half_mask_width = mask_width // 2
    ratio = 1/scale
    margin = int(half_mask_width - np.floor(half_mask_width * ratio))
    mask[:, :, margin:-margin, margin:-margin] = 1
    return mask

def fix_mask_with_scale_matrix(mask, scale_matrix):
    scale = scale_matrix[0][0]
    if scale <= 1:
        mask = torch.ones_like(mask)
        return mask
    else:
        mask = torch.zeros_like(mask)
        mask_width = mask.shape[2]
        half_mask_width = mask_width // 2
        ratio = 1/scale
        margin = int(half_mask_width - np.floor(half_mask_width * ratio.cpu()))
        mask[:, :, margin:-margin, margin:-margin] = 1
        return mask


class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2,
                 pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0,
                 blur_fade_kimg=0, added_kwargs=None):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

        self.added_kwargs = added_kwargs
        self.training_mode = self.G.training_mode
        if self.added_kwargs is not None:
            if self.added_kwargs.consistency_mode == 'multiscale_forward' or self.added_kwargs.consistency_mode == 'multiscale_inverse':
                self.loss_l1 = losses.Masked_L1_Loss().to(device)
                if self.added_kwargs.consistency_loss_select == 'both':
                    self.loss_lpips = losses.Masked_LPIPS_Loss(net='alex', device=device)
                    util.set_requires_grad(False, self.loss_lpips)

    def style_mix(self, z, c, ws):
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        return ws

    def run_G(self, z, c, transform, update_emas=False):
        mapped_scale = None
        crop_fn = None
        if 'global' in self.training_mode:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            ws = self.style_mix(z, c, ws)
            assert(transform is None)
            img = self.G.synthesis(ws, transform=transform, update_emas=update_emas)
        elif 'multiscale' in self.training_mode:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            scale, mapped_scale = patch_util.compute_scale_inputs(self.G, ws, transform, c)
            ws = self.style_mix(z, c, ws)
            img = self.G.synthesis(ws, mapped_scale=mapped_scale, transform=transform, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, transform, gen_z,
                             gen_c, gain, cur_nimg, min_scale, max_scale):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, transform)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Scale/G/min_scale', min_scale)
                training_stats.report('Scale/G/max_scale', max_scale)
                
                # multiscale consistency fork
                if (self.added_kwargs.consistency_mode == 'multiscale_forward' or self.added_kwargs.consistency_mode == 'multiscale_inverse') and self.added_kwargs.consistency_lambda > 0:
                    
                    batch_size = transform.shape[0]

                    consistency_scale_min = self.added_kwargs.consistency_scale_min
                    consistency_scale_max = self.added_kwargs.consistency_scale_max
                    scaling_factors = torch.rand(batch_size) * (consistency_scale_max - consistency_scale_min) + consistency_scale_min

                    # allowing scale for the scale consistency reference to go out of bound (zoom out more than the coarsest scale)
                    if(self.added_kwargs.consistency_scale_sampling == 'unbounded'):
                        pass
                    # beta distribution mentioned in the paper
                    elif(self.added_kwargs.consistency_scale_sampling == 'beta'):
                        original_scale_factors = transform[:,0,0]
                        original_scale_factors_log = torch.log2(1/original_scale_factors)
                        n_mags = torch.clamp(original_scale_factors_log, min=1)
                        mode = 1/n_mags
                        mode_reduce = 1/(n_mags/torch.sqrt(n_mags))
                        alpha = torch.sqrt(1/mode_reduce)
                        beta = (alpha - 1 - mode * alpha + mode * 2)/mode
                        scaling_factors = torch.distributions.beta.Beta(alpha, beta).sample()
                        scaling_factors = 2**(scaling_factors * original_scale_factors_log).cpu()
                        pass
                    # scale is sampled uniformly
                    elif(self.added_kwargs.consistency_scale_sampling == 'uniform'):
                        original_scale_factors = transform[:,0,0].cpu()
                        original_scale_factors_log = torch.log2(1/original_scale_factors)
                        consistency_scale_min_log = np.log2(consistency_scale_min)
                        consistency_scale_max_log = np.log2(consistency_scale_max)
                        normalization_min = torch.clamp(original_scale_factors_log, max=consistency_scale_min_log)
                        normalization_max = torch.clamp(original_scale_factors_log, max=consistency_scale_max_log) - normalization_min
                        alpha = beta = torch.ones_like(original_scale_factors)
                        scaling_factors_pre = torch.distributions.beta.Beta(alpha, beta).sample()
                        scaling_factors = 2**(scaling_factors_pre * normalization_max + normalization_min).cpu()
                        pass
                    

                    scaling_matrices = isotropic_scaling_matrix_2D_torch(scaling_factors).to(self.device)
                    scaling_matrices_inverted = isotropic_scaling_matrix_2D_torch(1/scaling_factors).to(self.device)
                    transform_mod = torch.bmm(transform, scaling_matrices)

                    teacher_c = torch.log2(1/transform_mod[:,0,0]).unsqueeze(1)
                    teacher_img, _gen_ws = self.run_G(gen_z, teacher_c, transform_mod)

                    if self.added_kwargs.random_gradient_off:
                        disbale_student = bool(random.getrandbits(1))
                        if disbale_student:
                            gen_img = gen_img.detach()
                        else:
                            teacher_img = teacher_img.detach()

                    # forward scale consistency
                    if self.added_kwargs.consistency_mode == 'multiscale_forward':
                        #computing L1
                        teacher_crop, teacher_mask = apply_affine_batch_and_fix_mask(teacher_img, scaling_matrices_inverted)
                        l1_loss = self.loss_l1(gen_img, teacher_crop, teacher_mask[:, :1])                        
                        teacher_loss = l1_loss[:, None] 
                        
                        #computing LPIPS if chosen
                        if self.added_kwargs.consistency_loss_select == 'both':
                            lpips_loss = self.loss_lpips(losses.adaptive_downsample256(gen_img), losses.adaptive_downsample256(teacher_crop), losses.adaptive_downsample256(teacher_mask[:, :1], mode='nearest')                    )
                            teacher_loss = (l1_loss + lpips_loss)[:, None]                                                             
                            training_stats.report('Loss/G/loss_teacher_lpips', lpips_loss)

                    # inverse scale consistency
                    elif self.added_kwargs.consistency_mode == 'multiscale_inverse':
                        #computing L1
                        gen_crop, gen_mask = apply_simplified_affine_batch_and_fix_mask(gen_img, scaling_matrices)
                        teacher_img = teacher_img * gen_mask
                        gen_crop = gen_crop * gen_mask
                        l1_loss = self.loss_l1(gen_crop, teacher_img, gen_mask[:, :1])
                        teacher_loss = l1_loss[:, None]

                        #computing LPIPS if chosen
                        if self.added_kwargs.consistency_loss_select == 'both':
                            lpips_loss = self.loss_lpips(losses.adaptive_downsample256(gen_crop), losses.adaptive_downsample256(teacher_img), losses.adaptive_downsample256(gen_mask[:, :1], mode='nearest')                    )
                            teacher_loss = (l1_loss + lpips_loss)[:, None]                                             
                            training_stats.report('Loss/G/loss_teacher_lpips', lpips_loss)

                    # computing final loss_Gmain value and reproting 
                    loss_Gmain = (loss_Gmain + self.added_kwargs.consistency_lambda * teacher_loss)   
                    training_stats.report('Loss/G/loss_teacher_l1', l1_loss)
                    training_stats.report('Loss/G/loss_total', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size],
                                             gen_c[:batch_size],
                                             transform[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, transform, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

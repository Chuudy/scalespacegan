# Moon dataset training
python ./scalescapcegan/train.py --cfg=stylegan3-ms --gpus=8 --batch=32 --gamma=2 --aug=ada --kimg=25000 --batch-gpu=4 \
--training_mode=multiscale --metrics=fid1k_full_multiscale_continuous_mix --snap=10 --cmax=256 \
--outdir ./runs/moon \
--auto-resume reconstruction6 \
--progfreqs --random_gradient_off \
--consistency_lambda=4 \
--warmup_kimgs=2000 --skewed_sampling=4000 --boost_first=4 --boost_last=4 --scale_count_fixed=5 \
--n_bins=2 --bin_transform_offset=3 --auto_last_redist_layer \
--data ./data/moon/sg3/continuous_96000_256_0-6.zip

# Spain dataset training
python ./scalescapcegan/train.py --cfg=stylegan3-ms --gpus=8 --batch=32 --gamma=2 --aug=ada --kimg=25000 --batch-gpu=4 \
--training_mode=multiscale --metrics=fid1k_full_multiscale_continuous_mix --snap=10 --cmax=384 \
--outdir ./runs/spain \
--auto-resume reconstruction8 \
--progfreqs --random_gradient_off \
--consistency_lambda=4 \
--warmup_kimgs=3000 --skewed_sampling=6000 --boost_first=8 --boost_last=4 --scale_count_fixed=5 \
--n_bins=3 --bin_transform_offset=3 --auto_last_redist_layer \
--data ./data/spain/sg3/continuous_156000_256_0-8.zip

# Himalayas dataset training
python ./scalescapcegan/train.py --cfg=stylegan3-ms --gpus=8 --batch=32 --gamma=2 --aug=ada --kimg=25000 --batch-gpu=4 \
--training_mode=multiscale --metrics=fid1k_full_multiscale_continuous_mix --snap=10 --cmax=384 \
--outdir ./runs/himalayas \
--auto-resume reconstruction8 \
--progfreqs --random_gradient_off \
--consistency_lambda=4 \
--warmup_kimgs=3000 --skewed_sampling=6000 --boost_first=4 --boost_last=4 --scale_count_fixed=5 \
--n_bins=3 --bin_transform_offset=3 --auto_last_redist_layer \
--data ./data/himalayas/sg3/continuous_156000_256_0-8.zip

# Rembrandt dataset training
python ./scalescapcegan/train.py --cfg=stylegan3-ms --gpus=8 --batch=32 --gamma=2 --aug=ada --kimg=25000 --batch-gpu=4 \
--training_mode=multiscale --metrics=fid1k_full_multiscale_continuous_mix --snap=10 --cmax=384 \
--outdir ./runs/rembrandt \
--auto-resume reconstruction8 \
--progfreqs --random_gradient_off \
--consistency_lambda=4 \
--warmup_kimgs=3000 --skewed_sampling=6000 --boost_first=4 --boost_last=4 --scale_count_fixed=5 \
--n_bins=3 --bin_transform_offset=3 --auto_last_redist_layer \
--data ./data/rembrandt/sg3/continuous_156000_256_0-8.zip
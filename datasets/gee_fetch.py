import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import ee
import geemap

import click
import numpy as np
import cv2
import numpy as np
import multiprocessing
import time
from scalespacegan.util.misc import save_image, load_image, sample_image, create_dir, print_same_line, parse_comma_separated_number_list
from scalespacegan import dnnlib
from glob import glob
import logging

logging.getLogger('earthengine').setLevel(logging.CRITICAL)
logging.getLogger('geemap').setLevel(logging.CRITICAL)
logging.getLogger('ee').setLevel(logging.CRITICAL)

#=======================================================


@click.command()

# Required.
@click.option('--output_dir',           help='Location were the data will be stored',    metavar='DIR',                 required=True    )

# Optional
@click.option('--n_processes',          help='Number of subsequent processes',           metavar='INT',                 default=12       ) # uses multiprocessing to download patches in parallel if greater than 1
@click.option('--batch_size',           help='Size of the multiprocessing batch',        metavar='INT',                 default=1000     ) # uses multiprocessing to download patches in parallel if greater than 1

# Optional for modifying default sampling
@click.option('--scales',               help='What scales should be sampled',            metavar='[NAME|A,B,C|none]',   type=parse_comma_separated_number_list  )
@click.option('--samples',              help='Sets number of samples per scale',         metavar='[NAME|A,B,C|none]',   type=parse_comma_separated_number_list  )

# Optional choice of the map regions
@click.option('--area',                 help='generator training mode',    type=click.Choice(['spain', 'himalayas']),   default="spain"  )


#=======================================================


def main(**kwargs):

    # Parse arguments
    opts = dnnlib.EasyDict(kwargs)
    root_dir = opts.output_dir
    n_processes = opts.n_processes
    batch_size = opts.batch_size
    selected_area = opts.area

    # Set aditional parameters
    dst_dir_name_base = "continuous"
    data_dir = os.path.join(root_dir, f"{selected_area}/samples")
    output_res = (256, 256) # describes size of the captured sample, however is resampled to 256x256 at the end - helps to remove some aliasing in captured footage
    tmp_dir = os.path.join(root_dir, f"{selected_area}/tmp_tifs")
    scale_multiplier = 215 if selected_area == "spain" else 217

    # Set number of sampels per scale
    selected_scales = [0, 1, 2, 3, 4, 5, 6, 7]
    sample_counts = [500, 1500, 4000, 30000, 30000, 30000, 30000, 30000]
    if opts.scales and opts.samples:
        if len(opts.scales) == len(opts.samples):
            selected_scales = opts.scales
            sample_counts = opts.samples
    total_sample_count = np.sum(np.array(sample_counts))
    
    #-----------------------

    dst_dir_name = dst_dir_name_base + "_" + str(total_sample_count) + "_" + str(output_res[0]) + "_" + str(selected_scales[0]) + "-" + str(selected_scales[-1]+1)
    scale_dir = os.path.join(data_dir, dst_dir_name)
    create_dir(scale_dir)
    create_dir(tmp_dir)
    
    base_scale = selected_scales[0]

    tmp_idx = 0
    curr_cummulative_samples = 0

    print(f"Starting processing", flush=True)
    for idx_scale, sample_count in zip(selected_scales, sample_counts):

        items = []
        current_scale = idx_scale

        for idx_img in range(sample_count):
            scale = np.random.uniform(low=current_scale, high=current_scale+1)
            items.append((tmp_idx, scale))
            tmp_idx += 1
        
        if n_processes <= 1:
            for (idx_img, scale) in items:            
                process_item(idx_img, scale, base_scale, scale_multiplier, output_res, scale_dir, tmp_dir, selected_area)
                print_same_line(f"Processed {idx_img+1} out of {total_sample_count} images.")
        else:
            print(f"\nProcessing patches for scale {current_scale}", flush=True)
            for i in range(0, sample_count, batch_size):
                batch_items = items[i:i+batch_size]
                while True:
                    pool = multiprocessing.Pool(n_processes)
                    results = pool.starmap(run_process, [(idx_img, scale, base_scale, scale_multiplier, output_res, scale_dir, tmp_dir, selected_area) for (idx_img, scale) in batch_items])
                    failed_indices = []
                    for result in results:
                        if isinstance(result, Exception):
                            print(str(result), flush=True)
                            failed_indices += [int(s) for s in str(result).split() if s.isdigit()]
                    if len(failed_indices) > 0:     
                        batch_items = [items[i-curr_cummulative_samples] for i in failed_indices]
                        print(failed_indices, flush=True)
                    else:
                        break
                
                print_same_line(f"{np.clip(i+batch_size, a_min=0, a_max=sample_count)} patches out of {int(sample_count)} for scale {current_scale} finished")
        
        curr_cummulative_samples += sample_count


#=======================================================


def run_process(idx_img, scale, base_scale, scale_multiplier, output_res, scale_dir, tmp_dir, selected_area):
    try:
        return process_item(idx_img, scale, base_scale, scale_multiplier, output_res, scale_dir, tmp_dir, selected_area)
    except Exception as e:
        return e


#=======================================================

def process_item(idx_img, scale, base_scale, scale_multiplier, output_res, scale_dir, tmp_dir, selected_area):
    
    if check_if_file_exists(idx_img, scale_dir):
        return True

    try:
        ee.Initialize()
    except:
        raise ValueError(f"EE cannot be initialized. Make sure you authenticated your google earth API.")

    if selected_area == "spain":
        image = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')\
        .filterDate('2020-01-01', '2022-01-31')\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 0.02))\
        .select('B4', 'B3', 'B2')\
        .mosaic()\
        .unitScale(300, 3000)
        x_min = -8.327637
        x_max = -0.944824
        y_min = 37.701207
        y_max = 43.181147
    elif selected_area == "himalayas":
        image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate('2017-01-01', '2022-01-31')\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
        .select('B4', 'B3', 'B2')\
        .median()\
        .unitScale(0, 4000)
        x_min = 80
        y_min = 27
        x_max = 86
        y_max = 32


    full_size = min(x_max - x_min, y_max - y_min)
    full_scale = full_size * scale_multiplier
    temp_file = os.path.join(tmp_dir, f"s_{scale-base_scale}_{idx_img}.tif")

    def download_patch(scale, temp_file):
        current_log_scale = scale
        current_scale = 2**current_log_scale

        size = full_size / current_scale
        gee_scale = full_scale / current_scale    

        # sample random patch
        xx = float(np.random.uniform(x_min, x_max - size, 1)[0])
        yy = float(np.random.uniform(y_min, y_max - size, 1)[0])

        sample_polygon = ee.Geometry.Polygon(
            [
                [
                    [xx, yy],
                    [xx, yy+size],
                    [xx+size, yy+size],
                    [xx+size, yy],
                    [xx, yy],
                ]
            ],
            None,
            False,
        )

        # clip the selected image according to the region of interest
        image_sample = image.clip(sample_polygon).unmask()
        
        # download image
        with SuppressOutput():
            geemap.ee_export_image(image_sample, filename=temp_file, scale=gee_scale, region=sample_polygon, file_per_band=False)
        if(os.path.exists(temp_file)):
            return True, current_log_scale    
        else:
            return False, current_log_scale

    try:
        num_trials = 0
        while True: # Try downloading a few times if fails
            success, current_log_scale = download_patch(scale, temp_file)
            time.sleep(0.01)
            num_trials += 1
            if success:
                break

            if num_trials < 2:
                continue
            else:
                raise ValueError(f"Download failed after {num_trials} tries -- breaking  {idx_img}")
            
        if success: # if tiff file sucessfully saved it is converted to PNG and then removed
            img = load_image(temp_file, normalize=False)
            # print(img.shape)
            
            # color correction
            tmp_output_res = (512, 512)
            img = sample_image(img, 0, 0, *tmp_output_res)
            img = np.clip(img, 0, 1)            
            if "spain" in scale_dir:
                img = np.power(img, 1/1.8)
                img[..., 2] = np.power(img[..., 2], 1/0.85)

            img = cv2.resize(img, output_res, interpolation=cv2.INTER_AREA)

            # output
            out_file = os.path.join(scale_dir, f"{idx_img:07}_{current_log_scale-base_scale:.4f}.png")
            save_image(img, out_file, jpeg_quality=99)
            
            os.remove(temp_file)
    except:
        raise ValueError(f"Couldn't process the sample number  {idx_img}")


#=======================================================


def check_if_file_exists(idx, directory):
    expression = os.path.join(directory, f"{idx:07}*")
    matched_files = glob(expression)
    exists = len(matched_files) > 0 
    return exists


#=======================================================

class SuppressOutput:
    def __enter__(self):
        # Redirect stdout and stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

#=======================================================

if __name__ == "__main__":    
    main()

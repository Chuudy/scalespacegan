import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import click
import numpy as np
import cv2
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ChromeOptions
from PIL import Image
import numpy as np
from io import BytesIO
import multiprocessing
import time
from scalespacegan.util.misc import save_image, create_dir, print_same_line, parse_comma_separated_number_list
from scalespacegan import dnnlib
from glob import glob


#=======================================================


@click.command()

# Required.
@click.option('--output_dir',           help='Location were the data will be stored',    metavar='DIR',                 required=True    )

# Optional
@click.option('--n_processes',          help='Number of subsequent processes',           metavar='INT',                 default=1        ) # uses multiprocessing to download patches in parallel if greater than 1
@click.option('--batch_size',           help='Size of the multiprocessing batch',        metavar='INT',                 default=1000     ) # uses multiprocessing to download patches in parallel if greater than 1

# Optional for modifying default sampling
@click.option('--scales',               help='What scales should be sampled',            metavar='[NAME|A,B,C|none]',   type=parse_comma_separated_number_list  )
@click.option('--samples',              help='Sets number of samples per scale',         metavar='[NAME|A,B,C|none]',   type=parse_comma_separated_number_list  )


#=======================================================


def main(**kwargs):

    # Install chrome driver
    chrome_driver_filepath = chromedriver_autoinstaller.install() 

    # Parse arguments
    opts = dnnlib.EasyDict(kwargs)
    root_dir = opts.output_dir
    n_processes = opts.n_processes
    batch_size = opts.batch_size

    # Set aditional parameters
    dst_dir_name_base = "continuous"
    browser_res = 1024
    data_dir = os.path.join(root_dir, "rembrandt/samples")
    output_res = (256, 256)
    darkness_thres = 0.1*255 

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
        
        if n_processes < 1:
            for (idx_img, scale) in items:            
                process_item(idx_img, scale, base_scale, browser_res, darkness_thres, output_res, scale_dir, chrome_driver_filepath)
                print_same_line(f"Processed {idx_img+1} out of {total_sample_count} images.")
        else:
            print(f"\nProcessing patches for scale {current_scale}", flush=True)
            for i in range(0, sample_count, batch_size):
                batch_items = items[i:i+batch_size]
                while True:
                    pool = multiprocessing.Pool(n_processes)
                    results = pool.starmap(run_process, [(idx_img, scale, base_scale, browser_res, darkness_thres, output_res, scale_dir, chrome_driver_filepath) for (idx_img, scale) in batch_items])
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


def run_process(idx_img, scale, base_scale, browser_res, darkness_thres, output_res, scale_dir, chrome_driver_filepath):
    try:
        return process_item(idx_img, scale, base_scale, browser_res, darkness_thres, output_res, scale_dir, chrome_driver_filepath)
    except Exception as e:
        return e


#=======================================================

def check_image_quality(img, size, threshold):
    # Collect crops from the four corners
    crops = []
    crops.append(img[:size, :size, :])
    crops.append(img[:size, -size:, :])
    crops.append(img[size:, :size, :])
    crops.append(img[-size:, -size:, :])

    # Calculate the mean value for each crop
    means = [np.mean(crop) for crop in crops]

    # Check if any mean value is below the threshold
    for mean in means:
        if mean < threshold:
            return False

    # All mean values are above the threshold
    return True

#=======================================================

def process_item(idx_img, scale, base_scale, browser_res, darkness_thres, output_res, scale_dir, chrome_driver_filepath):
    
    if check_if_file_exists(idx_img, scale_dir):
        return True

    try:
        opts = ChromeOptions()
        opts.set_capability("goog:loggingPrefs", { "browser": "ALL"})
        opts.add_argument('--headless')
        opts.add_argument('--no-sandbox')
        opts.add_experimental_option('excludeSwitches', ['enable-logging'])
        service = Service(chrome_driver_filepath)
        driver = webdriver.Chrome(options=opts, service=service)
        driver.set_window_size(browser_res, browser_res)
    except:
        raise ValueError(f"Driver error: Couldn't create a driver  {idx_img}")

    img_space_scale = 1./(2**(scale))

    np.random.seed(idx_img)

    # specifies the region that should be captured
    min_pos = np.array([0.025, 0.025])
    max_pos = np.array([1 - min_pos[0] - img_space_scale, 1 - min_pos[1] - img_space_scale - 0.16])
    
    sample = np.random.uniform(size=2)
    pos = sample * (max_pos - min_pos) + min_pos

    url = f"https://hyper-resolution.org/view.html?pointer=0.346,0.964&r=\
            {pos[0]:.4f},\
            {pos[1]:.4f},\
            {img_space_scale:.4f},\
            {img_space_scale:.4f},\
            &i=Rijksmuseum/SK-C-5/SK-C-5_VIS_5-um_2020-09-08"
    
    imgs = []
    for i in range(3): # image is acquired 3 times to exclude occasional captures of central part of the image
        try:
            driver.get(url)
        except:        
            close_driver(driver)
            raise ValueError(f"Driver error: Couldn't get the url  {idx_img}")
        
        ready = True

        time.sleep(4) # wait some time so the image can load fully still loading
        
        try:
            screenshot_png = driver.get_screenshot_as_png()
        except:        
            close_driver(driver)
            raise ValueError(f"Driver error: Couldn't get the screenshot {idx_img}")

        # Convert the screenshot to a PIL Image
        img = Image.open(BytesIO(screenshot_png))
        img = np.array(img)
        
        height, width, depth = img.shape
        start_x = int((width - 512) / 2)
        start_y = int((height - 512) / 2)
        img = img[start_y:start_y+512, start_x:start_x+512, :]
        
        # resize to output resolution
        img = cv2.resize(img, output_res, interpolation=cv2.INTER_AREA)
        imgs.append(img)

    if np.sum((imgs[0] - imgs[1])) != 0 or np.sum((imgs[1] - imgs[2])) != 0: # checking if all images are the same
        raise ValueError(f"One of the captured images is incorrect  {idx_img}")
    else:
        ready = True

    current_scale = scale - base_scale
    out_path = os.path.join(scale_dir, f"{idx_img:07}_{current_scale:.4f}.png")
    save_image(img, out_path)
    
    close_driver(driver)

    return ready


#=======================================================


def close_driver(driver):
    driver.close()
    driver.quit()


#=======================================================


def check_if_file_exists(idx, directory):
    expression = os.path.join(directory, f"{idx:07}*")
    matched_files = glob(expression)
    exists = len(matched_files) > 0 
    return exists


#=======================================================


if __name__ == "__main__":    
    main()

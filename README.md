# ScaleSpace-GAN
[Project Page](https://scalespacegan.mpi-inf.mpg.de/) | [Paper](https://scalespacegan.mpi-inf.mpg.de/files/scalespacegan_paper.pdf) | [Bibtex](https://scalespacegan.mpi-inf.mpg.de/files/scalespacegan.bib)

Learning Images Across Scales Using Adversarial Training.\
[Krzysztof Wolski](https://people.mpi-inf.mpg.de/~kwolski/), [Adarsh Djeacoumar](https://scholar.google.com/citations?user=3oeUgGEAAAAJ&hl=en), [ Alireza Javanmardi](https://av.dfki.de/members/javanmardi/), [Hans-Peter Seidel](https://people.mpi-inf.mpg.de/~hpseidel/), [Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/), [Guillaume Cordonnier](https://www-sop.inria.fr/members/Guillaume.Cordonnier/), [Karol Myszkowski](https://people.mpi-inf.mpg.de/~karol/), [George Drettakis](https://www-sop.inria.fr/members/George.Drettakis/), [Xingang Pan](https://xingangpan.github.io/), [Thomas Leimkühler](https://people.mpi-inf.mpg.de/~tleimkue/)


**Table of Contents:**<br>
1. [Setup](#setup) - clone repository and set up conda environment
2. [Pretrained Models](#pretrained) - quickstart with pretrained models<br>
3. [Datasets](#datasets) - how to download datasets from the paper<br>
3. [Training](#training) - pipeline for training<br>
4. [Evaluation](#evaluation) - evaluation script<br>


<a name="setup"/>

## Setup

Clone this repo and create conda environment:

```bash
git clone https://github.com/Chuudy/scalespacegan.git
cd scalespacegan
conda env create -f environment.yml
conda activate scalespacegan
```

Alternatively, to create Conda environemnt that uses Pytorch 2.3 execute following commands:

```bash
git clone https://github.com/Chuudy/scalespacegan.git
cd scalespacegan
conda env create -f environment_torch2.yml
conda activate scalespaceganT2
```

For all the next steps make sure you're in the root directory of cloned repository.


<a name="pretrained"/>

## Quickstart with pretrained models

Pretrained models can be downlaoded from [this repository](https://scalespacegan.mpi-inf.mpg.de/files/models/) or by using [download_models.sh](./models/download_models.sh) script located in the models directory. Script can be run using following lines:

```bash
chmod +x models/download_models.sh
./models/download_models.sh
```

If you want to visualize the model in real time you can use script [visualizer.py](./scalespacegan/visualizer.py).
Script can be used as follows:

```bash
python ./scalespacegan/visualizer.py --model_dir=<directory_of_the_model> --model_file=<model_file>
```

If you used the download script and you want to run e.g., the moon model, execute following line:

```bash
python ./scalespacegan/visualizer.py --model_dir=./models --model_file=moon_rec6.pkl
```

In addition to real-time visualizer we provide script [synthesizer.py](./scalespacegan/synthesizer.py) that can produce an image of any arbitrary resolution.
The script can be used as follows:

```bash
python ./scalespacegan/synthesizer.py --model_dir=./models --model_file=moon_rec6.pkl --output_dir=./imgs --resolution=512
```


<a name="datasets"/>

## Datasets

Currently three datasets are available to download using script provided in [datasets](./datasets) directory.
These script were tested on Windows 11 machine, however should work also with Linux distributions.
All these script use multiprocessing by default. Number of parallel process can be changed by modifying argument `n_processes`.
For Moon and Rembrandt dataset it is required to have Chrome web browser isntalled.

### Datasets Downloading

#### Moon dataset

Script for downloading Moon dataset uses chromium library (included in the environemnt.yml file).
To download the dataset run following line with `scalespacegan` conda environment active:

```bash
python ./datasets/moon_fetch.py --output_dir=./data --n_processes=12 --batch_size=1000
```

#### Rembrandt dataset

Script for downloading Rembrandt dataset uses chromium library (included in the environemnt.yml file).
To download the dataset run following line with `scalespacegan` conda environment active:

```bash
python ./datasets/rembrandt_fetch.py --output_dir=./data --n_processes=12 --batch_size=1000
```

#### Spain and Himalayas dataset

Script for downloading Rembrandt dataset uses geemap library (included in the environemnt.yml file) adn requires google earth account.
To setup all the prerequisites execute the Jupyter Notebook [gee_init.ipynb](/datasets/gee_init.ipynb)

Once setup is finished execute following line for Spain dataset:

```bash
python ./datasets/gee_fetch.py --output_dir=./data --n_processes=12 --batch_size=1000
```

For Himalayas dataset:

```bash
python ./datasets/gee_fetch.py --output_dir=./data --n_processes=12 --batch_size=1000 --area=himalayas
```

### Datasets Preprocessing

Once the datasets are donwloaded they still need to be processed in order to extract the labels and create the final zip file. For this purpose script [dataset_tool_multiscale_continuous.py](./scalespacegan/dataset_tool_multiscale_continuous.py) shoudl be used as follows:

```bash
python ./scalespacegan/dataset_tool_multiscale_continuous.py \
--source=<directory_with_the_patches> \
--dest=<output_directory> \
--resolution=256
```

If you followed the exact steps of downloading the data, the moon dataset can be processed as follows:

```bash
python ./scalespacegan/dataset_tool_multiscale_continuous.py \
--source=./data/moon/samples/continuous_96000_256_0-6 \
--dest=./data/moon/sg3/continuous_96000_256_0-6.zip \
--resolution=256x256
```


<a name="training"/>

## Training

If you followed the exact steps of zipping the data, the training for the moon dataset can be performed by executing following line:

```bash
python ./scalespacegan/train.py --cfg=stylegan3-ms --gpus=8 --batch=32 --gamma=2 --aug=ada --kimg=25000 --batch-gpu=4 \
--training_mode=multiscale --metrics=fid1k_full_multiscale_continuous_mix --snap=10 --cmax=256 \
--outdir ./runs/moon \
--auto-resume reconstruction6 \
--progfreqs --random_gradient_off \
--consistency_lambda=4 \
--warmup_kimgs=2000 --skewed_sampling=4000 --boost_first=4 --boost_last=4 --scale_count_fixed=5 \
--n_bins=2 --bin_transform_offset=3 --auto_last_redist_layer \
--data ./data/moon/sg3/continuous_96000_256_0-6.zip
```

See the script [train.sh](./train.sh) for more training examples.

Training notes:
- arguments `--batch-gpu` and `--gamma` are taken from Stylegan 3 [recommended configurations](https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md#recommended-configurations).
- argument `--gpus` specifies how many GPUs should be used in parallel.
- argument `--cmax` decides how many input fourier features is used during the training (number_of_features = 4*cmax). We use values of 256 for 6 scale magnitudes and 384 for 8 scale mangitudes.
- argument `--auto-resume` sepcifies name of the leaf directory in which all the trainign data will be stored. If this parameters is used the training will automatically resume from the last checkpoint if the same code is executed again.
- argument `--random_gradient_off` enables randomization of grandient backpropagation through one of the branches in scale-consistency loss.
- argument `--consistency_lambda` set weighting for the scale consistency loss.
- argument `--warmup_kimgs` specifies for how many iterations the scale distibution will transition from the one emphasizing coarse scales to uniform one.
- argument `--skewed_sampling` enables progressive scale sampling and specifies total number of iterationduring which distirbution changes from emphasis of the coarse scale to the emphasis of the fine scales.
- arguments `--boost_first` and `--boost_last` specifies scaling of how much more important are coarse scales at the beginning of the training and fine scales at the end of the training respectively.
- arguments `--n_bins` and `--bin_transform_offset` specify how many fourier feature bins are used and what is the scale offset between them (expressed in a log scale), respoectively.
- argument `--metrics` specifies which metrics should be computed at each snapshot. By default this argument is set to `fid1k_full_multiscale_continuous_mix` which takes minimal amount of time, but allows to track progress of the training.



<a name="evaluation"/>

## Evaluations

The new FID metrics described in the paper can be computed in a stanfalone mode using [calc_multiscale_metrics.py](./scalespacegan/calc_multiscale_metrics.py) script.

- For multiscale FID reported in the Tables 1, 2, and 3 of the paper you can use following options for the `--metrics` argument:  `fid1k_full_multiscale_continuous_mix`, `fid10k_full_multiscale_continuous_mix`, and `fid50k_full_multiscale_continuous_mix`, which computes multiscale FID for 1K, 10K and 50K patches respectively.
- To compute per scale bin multiscale FID that is demonstarted in Fig. 9 in the paper use following options for the `--metrics` argument: `fid1k_full_multiscale_continuous`, `fid10k_full_multiscale_continuous`, and `fid50k_full_multiscale_continuous`, which computes multiscale FID for 1K, 10K and 50K patches per scale bin respectively.

Here's an example how to compute the `fid50k_full_multiscale_continuous_mix` metric for the moon dataset trained by executing the example code from the previous section (the code assumes the model has been trained for 10000 kimgs):


```bash
python ./scalespacegan/calc_multiscale_metrics.py \
--network ./runs/moon/reconstruction6-stylegan3-ms-continuous_96000_256_0-6-gpus8-batch32-gamma2/network-snapshot-010000.pkl \
--data ./data/moon/sg3/continuous_96000_256_0-6.zip \
--metrics fid50k_full_multiscale_continuous_mix \
--gpus 1
```

If you get any errors at this point, please revise the exact name of the directory where the model is stored, as the training script will set its name based on the configuration you used.


## Acknowledgements

Our code is largely based on the [Anyres-GAN](https://github.com/chail/anyres-gan) repository ([license](./scalespacegan/LICENSE_anyresgan.txt)). Changes to Anyres-GAN code are documented in [diff](./scalespacegan/diff.txt).

<a name="citation"/>

## Citation
If you use this code for your research, please cite our paper:
```
@article{wolski2024scalespacegan,
	author = {Wolski, Krzysztof and Djeacoumar, Adarsh and Javanmardi, Alireza and Seidel, Hans-Peter and Theobalt, Christian and Cordonnier, Guillaume and Myszkowski, Karol and Drettakis, George and Pan, Xingang and Leimk\"{u}hler, Thomas},
	title = {Learning Images Across Scales Using Adversarial Training},
	year = {2024},
	issue_date = {July 2024},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {43},
	number = {4},
	issn = {0730-0301},
	url = {https://doi.org/10.1145/3658190},
	doi = {10.1145/3658190},
	journal = {ACM Trans. Graph.},
	month = {jul},
	articleno = {131},
	numpages = {13}
}
```


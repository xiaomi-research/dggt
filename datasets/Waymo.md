# Preparing Waymo Dataset
> Note: This document is modified from [OmniRe](https://github.com/ziyc/drivestudio/blob/main/docs/Waymo.md) and [GaussianSTORM] https://github.com/NVlabs/GaussianSTORM/docs/WAYMO.md

## 1. Register on Waymo Open Dataset

#### Sign Up for a Waymo Open Dataset Account and Install gcloud SDK

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details.

#### Set Up the Data Directory

Once you've registered and installed the gcloud SDK, create a directory to house the raw data:

```bash
# create the data directory or create a symbolic link to the data directory
mkdir -p ./data/waymo/raw   
```

## 2. Environment

We highly recommend setting up another environment for data processing as the TensorFlow dependencies often conflict with our main environment.
```bash
conda create -n dggt_data python=3.10
conda activate dggt_data
pip install -r requirements_data.txt
```

## 3. Download the Raw Data
For the Waymo Open Dataset, we first organize the scene names alphabetically and store them in `data/waymo_val_list.txt`. The scene index is then determined by the line number minus one.

For example, you can download 3 sequences from the dataset by:

```bash
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw/validation \
    --split_file  data/waymo_val_list.txt \
    --scene_ids 8 23 150
```
If you wish to run experiments on different scenes, please specify your own list of scenes.

You can also omit the `scene_ids` to download all scenes specified in the `split_file`:

```bash
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw/validation \
    --split_file data/dataset_scene_list/waymo_val_list.txt
```

<details>
<summary>If this script doesn't work due to network issues, consider manual download:</summary>

Download the [scene flow version](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) of Waymo to `./data/waymo/raw/validation`.

![Waymo Dataset Download Page](https://github.com/user-attachments/assets/a1737699-e792-4fa0-bb68-0ab1813f1088)

> **Note**: Ensure you're downloading the scene flow version to avoid errors. You can also download other versions of the Waymo dataset. Just make sure to comment out the relevant computational logic during inference.

</details>

## 4.1 Preprocess the Data for Training
Code for this section is coming soon!​

## 4.2 Preprocess the Data for Inference/test (Quick Start)
After downloading the raw dataset, you'll need to preprocess this compressed data to extract and organize various components.

#### Run the Preprocessing Script
To preprocess specific scenes of the dataset, use the following command:
```bash
python datasets/preprocess_waymo.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_list_file data/dataset_scene_list/waymo_train_list.txt \
    --scene_ids 700 754 23 \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save data/annotations/waymo 
```
Alternatively, preprocess a batch of scenes by providing the split file:
```bash
python datasets/preprocess_waymo.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split validation \
    --scene_list_file data/waymo_val_list.txt \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save data/annotations/waymo 
```
The extracted data will be stored in the `data/waymo/processed` directory.


## 5. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Install `SegFormer` (Skip if already installed)

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a seperate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint` .


#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/path/to/segformer

python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/validation \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/waymo_example_scenes.txt \
    --process_dynamic_mask
```
Replace `/pathtosegformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 6. Data Structure
After completing all preprocessing steps, the project files should be organized according to the following structure:
```bash
ProjectPath/data/
  └── waymo/
    ├── raw/validation
    │    ├── segment-454855130179746819_4580_000_4600_000_with_camera_labels.tfrecord
    │    └── ...
    └── processed/
         └──validation/
              ├── 000/
              │  ├──extrinsics/         # camera-to-ego (camera_to_ego) transformations: {cam_id}.txt
              │  ├──intrinsics/         # camera intrinsics: {cam_id}.txt
              │  ├──ego_pose/           # ego-vehicle to world transformations (4x4): {timestep:03d}.txt
              │  ├──depth_flows_4/      # downsampled (1/4) depth flow maps: {timestep:03d}_{cam_id}.npy
              │  ├──dynamic_masks/      # bounding-box-generated dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──ground_label_4/     # downsampled (1/4) ground labels extracted from point cloud, used for flow evaluation only: {timestep:03d}.txt
              │  ├──images/             # original camera images: {timestep:03d}_{cam_id}.jpg
              │  ├──images_4/           # downsampled (1/4) camera images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # lidar data: {timestep:03d}.bin
              │  ├──sky_masks/          # sky masks: {timestep:03d}_{cam_id}.png
              ├── 001/
              ├── ...
```
<div align="center">

# DGGT ：FEEDFORWARD 4D RECONSTRUCTION OF DYNAMIC DRIVING SCENES USING UNPOSED IMAGES
<a href="https://arxiv.org/abs/2512.03004" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv">
</a>
<a href="https://xiaomi-research.github.io/dggt/" target="_blank">
  <img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project Page">
</a>

**Xiaoxue Chen**¹,²*, **Ziyi Xiong**¹,²*, **Yuantao Chen**¹, **Gen Li**¹, **Nan Wang**¹,  
**Hongcheng Luo**², **Long Chen**², **Haiyang Sun**²†, **Bing Wang**², **Guang Chen**², **Hangjun Ye**²,✉,  
**Hongyang Li**³, **Ya-Qin Zhang**¹, **Hao Zhao**¹,⁴,✉

¹ AIR, Tsinghua University  
² Xiaomi EV  
³ The University of Hong Kong  
⁴ Beijing Academy of Artificial Intelligence  

\* These authors contributed equally
† Project leader

</div>



## Abstract

Our method introduces a fully pose-free feedforward framework **DGGT** for reconstructing dynamic driving scenes directly from unposed RGB images. The model predicts camera poses, 3D Gaussian maps, dynamic motion in a single pass — without per-scene optimization or camera calibration.

<details><summary>CLICK for the full abstract</summary>

> Autonomous driving needs fast, scalable 4D reconstruction and re-simulation for training and evaluation, yet most methods for dynamic driving scenes still rely on per-scene optimization, known camera calibration, or short frame windows, making them slow and impractical. We revisit this problem from a feedforward perspective and introduce **Driving Gaussian Grounded Transformer (DGGT)**, a unified framework for pose-free dynamic scene reconstruction. We note that the existing formulations, treating camera pose as a required input, limit flexibility and scalability. Instead, we reformulate pose as an output of the model, enabling reconstruction directly from sparse, unposed images and supporting an arbitrary number of views for long sequences. Our approach jointly predicts per-frame 3D Gaussian maps and camera parameters, disentangles dynamics with a lightweight dynamic head, and preserves temporal consistency with a lifespan head that modulates visibility over time. A diffusion-based rendering refinement further reduces motion/interpolation artifacts and improves novel-view quality under sparse inputs. The result is a single-pass, pose-free algorithm that achieves state-of-the-art performance and speed. Trained and evaluated on large-scale driving benchmarks (Waymo, nuScenes, Argoverse2), our method outperforms prior work both when trained on each dataset and in zero-shot transfer across datasets, and it scales well as the number of input frames increases.
</details>

## 🚧 Todo

- [√] Release pre-trained checkpoints on  Waymo, NuScenes and Argoverse2
- [√] Release the inference code of our model to facilitate further research and reproducibility.
- [ ] Release the training code [after paper accepted]


### 🚗 Dataset Support
This codebase provides support for Waymo Open Dataset, Nuscenes and Argoverse2. We provide instructions and scripts on how to download and preprocess these datasets:
| Dataset | Instruction |
|---------|-------------|
| Waymo | [Data Process Instruction](datasets/Waymo.md) |
| NuScenes | [Data Process Instruction](datasets/NuScenes.md) |
| Argoverse2 | [Data Process Instruction](datasets/ArgoVerse2.md) |


## Installation
### Installing dependencies

1. Create conda environment
```bash
conda create -n dggt python=3.10
conda activate dggt

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

2. Compile pointops2

```bash
cd third_party/pointops2
python setup.py install
cd ../..
```


### Downloading checkpoints
Download our pretrained inference model (trained on Waymo Open Dataset, 1 views) checkpoint [here](https://huggingface.co/xiaomi-research/dggt/resolve/main/model_latest_waymo.pt?download=true) to `pretrained/model_latest_waymo.pth`.

Download our pretrained diffusion model checkpoint [here](https://huggingface.co/xiaomi-research/dggt/resolve/main/model_difix.pkl?download=true) to `pretrained/diffusion_model.pth`.

Download TAPIP3D model checkpoint [here](https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth) to `pretrained/tracking_model.pth`.

Other checkpoints will be coming soon.
## Usage
### Quick start
You can test existing models on the Waymo Open dataset.
```bash
python inference.py \
    --image_dir /path/to/images \
    --scene_names 3 5 7 \
    --input_views 1 \
    --intervals 2 \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/checkpoint.pth \
    --output_path /path/to/output \
    -images \
    -depth \
    -diffusion \
    -metrics 
```
    --image_dir <path>: Specifies the directory containing the input images (required).
    --scene_names <names>: A string representing the scene names to process, supporting formats like 3 5 7 or "(3,7)" (required).
    --mode <mode>: Specifies the processing mode, with acceptable values of 1--train, 2--reconstruction, or 3--interplation (required).
    --ckpt_path <path>: The path to the pre-trained model weights file (required).
    --output_path <path>: The directory where the output results will be saved (required).
    --input_views <views>: Number of input cameras like 1 or 3(required).
    --intervals <interval>: The interval of interpolation frames when performing frame interpolation (mode=3), defaulting to 2 (optional).
    --sequence_length <length>: Defines the number of input frames to consider for each inference, defaulting to 4 (optional).
    --start_idx <index>: Indicates the starting index of the frames to process, defaulting to 0 (optional).
    -images: A flag that, when specified, enables the output of rendered images for each frame (optional).
    -depth: A flag that, when specified, enables the output of depth maps in .npy format for each frame (optional).
    -metrics: A flag that, when specified, enables the output of evaluation metrics (PSNR, SSIM, LPIPS) after processing (optional).
    -diffusion: Whether to use diffusion model to optimize the rendered images (time-consuming) (optional).


### Zero-shot and trained experiment​s

Quantitative Comparison under Trained and Zero-Shot Settings on nuScenes and Argoverse2 datasets. 

You can evaluate the model in two complementary settings to demonstrate both generalization and adaptability:

#### Zero-shot (Generalization)
You can use the model trained on Waymo to perform inference directly on the Argoverse2 or nuScenes datasets — without any retraining or pose calibration.

This setting highlights the model’s strong cross-dataset generalization and robustness to unseen driving domains.

Argoverse2/Nuscenes
```bash
python inference.py \
    --image_dir /path/to/argoverse_or_nuscenes_images \
    --scene_names 3 5 7 \
    --input_views 1 \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/waymo_checkpoint.pth \
    --output_path /path/to/output \
    -images \
    -depth \
    -metrics \
```

#### Trained (Adaptability / Upper-bound Performance)
You can also train the model on the target dataset (e.g., Argoverse2) and evaluate it on the same domain.

This setting measures the model’s in-domain adaptability, showing its capacity to achieve state-of-the-art reconstruction quality when optimized for the target environment.

Argoverse2/Nuscenes
```bash
python inference.py \
    --image_dir /path/to/argoverse_or_nuscenes_images \
    --scene_names 3 5 7 \
    --input_views 1 \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/argoverse_or_nuscenes_checkpoint.pth \
    --output_path /path/to/output \
```

Together, these two experiments verify that our model not only generalizes well across unseen scenes, but also scales effectively to achieve top performance when fine-tuned on new domains.
## Citation
If you find this project useful, please consider citing:

```
@article{chenfeedforward,
  title={Feedforward 4D Reconstruction for Dynamic Driving Scenes using Unposed Images},
  author={Chen, Xiaoxue and Xiong, Ziyi and Chen, Yuantao and Li, Gen and Wang, Nan and Luo, Hongcheng and Chen, Long and Sun, Haiyang and WANG, BING and Chen, Guang and others}
}
```

## License
This project is licensed under the Apache License 2.0.
Some files in this repository are derived from VGGT (facebookresearch/vggt) and are licensed under the VGGT upstream license. See NOTICE for details.

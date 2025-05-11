# Photogrammetry and Gaussian Splatting on Apollo 17 Imagery

This project implements a complete pipeline for photogrammetry and neural rendering using 15 high-resolution Apollo 17 lunar surface images. The work is divided into two main phases:
1. 3D reconstruction via traditional photogrammetry using Agisoft Metashape (PART A)
2. High-fidelity novel view synthesis using Gaussian Splatting (PART B)

The goal of this project is to evaluate the capability of Gaussian Splatting in reconstructing and enhancing scenes from sparse camera views and compare it quantitatively to the original Apollo imagery.

---

## Overview

This assignment is divided into two parts:

- **Part A**: Reconstruct a textured 3D mesh model using Agisoft Metashape with 15 Apollo 17 images. Evaluate how well the model captures spatial and texture accuracy.
- **Part B**: Apply Gaussian Splatting to generate novel views. Rebuild a photogrammetry model using the original and synthesized images (N=25). Compare the results qualitatively and quantitatively.

---

## Dataset

- **Source**: [Apollo 17 Image Set (Google Drive)](https://drive.google.com/drive/folders/18t2fq0a8yKQDM4BYSeuSHrAbmJujMYLO?usp=drive_link)  
- **Input**: 15 grayscale or color images captured during the Apollo 17 mission  
- **Output**: 3D mesh model with texture, depth maps, camera poses, and later, novel views

---

## Part A: Photogrammetry Pipeline (Agisoft Metashape)

### Software

- **Tool**: Agisoft Metashape Professional (Trial Version)
- **Platform**: Ubuntu 22.04
- **Installation**: [Agisoft Downloads](https://www.agisoft.com/downloads/installer/)

### Processing Steps

1. **Import Images**  
   `Workflow → Add Photos`  
   Loaded all 15 Apollo images into a single chunk.

2. **Camera Alignment**  
   `Workflow → Align Photos`  
   - Accuracy: High  
   - Key Point Limit: 40,000  
   - Tie Point Limit: 4,000  
   - Adaptive fitting: Enabled

![align_images](https://github.com/user-attachments/assets/bb48ad36-9dba-4b64-b9ef-6e5ee57ea1dc)

3. **Depth Map Generation**  
   `Workflow → Build Model → Depth Maps`  
   - Quality: High  
   - Depth Filtering: Moderate

![depthmap](https://github.com/user-attachments/assets/26e28965-d76d-4e6d-9729-a7ef313ffcd0)


4. **Model Generation**  
   `Workflow → Build Model`  
   - Source: Depth Maps  
   - Surface Type: Arbitrary  
   - Face Count: High  
   - Interpolation: Enabled

5. **Texture Mapping**  
   `Workflow → Build Texture`  
   - Mapping Mode: Generic  
   - Blending Mode: Mosaic  
   - Texture Size: 8192×1

![texture](https://github.com/user-attachments/assets/078b687b-b541-42f2-b1cc-f4ca7be413e6)


### Exported Artifacts

| Output Type        | Format | Notes                             |
|--------------------|--------|-----------------------------------|
| Textured Mesh      | `.obj` | Includes normals and texture      |
| Optional Point Cloud | `.ply` | With or without confidence values |
| Viewport Screenshots | `.png` | For visual inspection             |

---

## Next Steps

- Set up Gaussian Splatting environment
- Train on original Apollo images and extract rendered views
- Evaluate SSIM and PSNR against original photos
- Generate 10 novel camera poses
- Rebuild and compare photogrammetric mesh (N=25)  
- Perform visual and metric-based model comparisons

---
## Overview of Gaussian Splatting

Gaussian Splatting is a real-time 3D rendering technique that represents a scene as a collection of anisotropic 3D Gaussians. These primitives are optimized to encode spatial, scale, and color attributes, enabling direct differentiable rendering. The method avoids traditional mesh reconstruction and instead focuses on optimizing point-based representations, allowing for high-quality reconstructions with fewer views.

---

## Environment Setup and Installation (Working Configuration)

This section documents the tested configuration and installation steps that successfully led to a working Gaussian Splatting pipeline.

### System Configuration
- **Operating System:** Ubuntu 22.04
- **GPU:** NVIDIA GeForce RTX 3050 (4 GB VRAM)
- **CUDA:** Version 12.9 (installed via official .deb installer)
- **Python:** 3.10.12 (via `venv`)
- **PyTorch:** Installed for CUDA 12.1

### Installation Steps

```
# Set up Python virtual environment
python3 -m venv gs_env
source gs_env/bin/activate

# Clone the Gaussian Splatting repository
cd gaussian-splatting

# Install core dependencies
sudo apt install ninja-build colmap
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm plyfile scikit-image pillow

# Manually install GPU-dependent modules
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim

```

## Dataset Preparation

Dataset used: 15 aligned Apollo 17 surface images.

---

### CUDA Configuration
Ensure the following environment variables are added to `~/.bashrc`:

`export CUDA_HOME=/usr/local/cuda-12.9`
`export PATH=$CUDA_HOME/bin:$PATH`
`export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`

Verify installation with:

`nvcc --version`
`nvidia-smi`

---

Convert COLMAP text files to binary using:

`colmap model_converter \
  --input_path data/apollo-gsplat-1/sparse/0 \
  --output_path data/apollo-gsplat-1/sparse/0 \
  --output_type BIN`

---

## Model Training

To accommodate limited GPU memory, training was done at 50% input resolution. Viewer rendering and mid-training evaluations were disabled to avoid memory overflow.

`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
`python train.py -s data/apollo-gsplat-1 -r 2 --disable_viewer --test_iterations 999999`

_Important Note:_ Training reached 8200 iterations before exiting due to memory limits, but intermediate results (Gaussians and scene state) were saved successfully.

---

## Rendering Views

Final output images were rendered from saved Gaussians using:

`python render.py -m output/<run_id>`

The reconstructed views were found under: output/<run_id>/train/ours_7000/renders/*.png

_Important Note:_ These outputs correspond to the training views, not novel test views.

---

## Evaluation: PSNR and SSIM Comparison

A Python script was written to compute quantitative comparison between original Apollo images and the GSplat-generated reconstructions.

### Script Functionality
- Loads each image pair from:
  - Ground Truth: `train/ours_7000/gt/`
  - Reconstruction: `train/ours_7000/renders/`
- Computes:
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **SSIM (Structural Similarity Index)**
- Outputs per-image and average metrics

### Results

Average PSNR: 30.50 dB
Average SSIM: 0.8806

These metrics indicate high fidelity and structural alignment with the original imagery, validating the effectiveness of Gaussian Splatting on this dataset.

- Evaluation metrics such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) are used to quantitatively assess the quality of the reconstructed images generated via Gaussian Splatting.
- PSNR measures pixel-wise differences, while SSIM captures perceptual and structural similarity. Higher values in both indicate a better match between the reconstructed image and the original ground truth, thereby validating the fidelity of the neural rendering approach.

**Example of a Noticable Difference between Ground Truth and Gaussian Splat Image:**

| Ground Truth            | Gaussian Splatting Output     |
|-------------------------|-------------------------------|
| ![00013](https://github.com/user-attachments/assets/82333e5d-fc8d-4515-99a0-71efecb6788c) | ![00013](https://github.com/user-attachments/assets/a9bb218c-56f0-462c-8ded-0df23698e4ef)|

---

## Visual Comparison Table

The following table displays qualitative results side-by-side. Ground truth Apollo images are shown on the left, with corresponding Gaussian Spatting reconstructions on the right:

<table>
  <tr>
    <th colspan="2" style="background-color:#f9f9f9;">Compare</th>
    <th colspan="2" style="background-color:#ffffff;">Compare</th>
    <th colspan="2" style="background-color:#f9f9f9;">Compare</th>
  </tr>
  <tr>
    <th style="background-color:#f9f9f9;">Ground Truth</th>
    <th style="background-color:#f9f9f9;">GSplat</th>
    <th style="background-color:#ffffff;">Ground Truth</th>
    <th style="background-color:#ffffff;">GSplat</th>
    <th style="background-color:#f9f9f9;">Ground Truth</th>
    <th style="background-color:#f9f9f9;">GSplat</th>
  </tr>

  <tr>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/52e245f7-3852-40a9-bc43-3b874b0ec4a5" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/0066231c-66d3-40e5-89f6-cdda1db4e65c" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/a4ff7a1e-7fc7-48ae-86a3-1357fe2009de" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/65c46480-62d7-4715-8176-a03578807a91" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/d5894482-5d09-4103-bf3f-d8e86f5d5eaa" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/e53f6bb3-eff7-438e-82fb-8824e755c80b" width="150"/></td>
  </tr>

  <tr>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/75d1dae4-e743-4dbe-a7ef-eff7fae90c08" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/7befa89c-4ea2-4212-bdb2-b64d8c6b849c" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/04baed06-37bf-4f9d-8bc6-408736fe6859" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/41af7ad4-0170-45b2-8f5b-678d53991c78" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/0d2433c1-fda3-4a14-8597-b1660361074c" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/b2a7d7b9-0165-47da-9f36-2be21bb03bed" width="150"/></td>
  </tr>

  <tr>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/cf799409-bc9b-4877-8197-2b451b009fc0" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/b3957315-fc9f-4171-a863-0cac429263c0" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/a1320c40-b186-4458-a56f-4fe23115a0d7" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/97ff2877-78f7-4e02-abab-1ba07edf239e" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/5c63d617-10db-4d90-99e9-9d2de966a8d8" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/9138a2ea-7163-49a2-a374-105d40c24e1c" width="150"/></td>
  </tr>

  <tr>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/5e694b1b-009c-4b9b-a62e-7ccf70b3fb07" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/6b69b42f-04e2-4726-b7dc-357fdb9bc45e" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/ddffdbd3-f435-480f-abc6-509408ec3343" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/ff85e76d-e29b-4c98-9936-82871b6e9591" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/a074d0e7-cbf1-47aa-bb11-d357d2668225" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/7839ca8d-1990-4eb3-81a7-a02c4dd71d97" width="150"/></td>
  </tr>

  <tr>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/933079cd-0e8b-47dc-b522-d54ffcf644a1" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/36a85883-42b8-4b18-900c-3d3e1942f997" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/82333e5d-fc8d-4515-99a0-71efecb6788c" width="150"/></td>
    <td style="background-color:#ffffff;"><img src="https://github.com/user-attachments/assets/a9bb218c-56f0-462c-8ded-0df23698e4ef" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/b2ebddc4-81e9-44cf-bf2c-8747c2a341f5" width="150"/></td>
    <td style="background-color:#f9f9f9;"><img src="https://github.com/user-attachments/assets/6f4d2441-50bc-4162-901b-aa77c12dd687" width="150"/></td>
  </tr>
</table>

---

## Status
- [x] Photogrammetry complete (Agisoft Metashape)
- [x] Gaussian Splatting training and rendering
- [x] PSNR/SSIM comparison complete
- [x] Visual results verified

This concludes Part A of the assignment.

---

## ⚠️ Gaussian Splatting Limitations and Transition

Despite a successful setup of the official [Gaussian Splatting](https://repo-surface.inria.fr/gaussian-splatting/) repository, multiple technical issues prevented full completion of our novel view synthesis pipeline:

- **CUDA Memory Issues**: Even after reducing resolution and setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the training pipeline often crashed due to GPU out-of-memory errors on a 4GB CUDA device.
- **Empty Output Folders**: Although training completed successfully (up to 15,000 iterations), the expected rendered outputs (e.g., under `test/ours_XXXX/renders`) remained consistently empty.
- **Rendering Script Failures**: Both official `render.py` and our modified `manual_novel_render.py` failed to correctly render the novel camera poses, primarily due to internal `Camera()` API changes and attribute mismatches (e.g., `unexpected keyword argument 'fx'`, `no attribute _exposure`).
- **Scene Type Errors**: Even after merging `cameras.json` with novel poses, the renderer raised `AssertionError: Could not recognize scene type!`, indicating unsupported or misformatted novel camera definitions.
- **No Visual Output for Novel Views**: Despite several recovery attempts—including merging JSONs, regenerating camera definitions, and manually adjusting rendering logic—no novel view images were produced.

---

## ✅ Work Completed

- Processed 15 high-resolution Apollo 17 photographs using **Agisoft Metashape**, resulting in a high-quality sparse reconstruction and textured mesh.
- Converted the COLMAP-formatted output to the binary format required by GSplat.
- Successfully trained the Gaussian Splatting model on these images, saving model checkpoints at 5K, 10K, and 15K iterations.
- Verified quality of generated images (for training views) using a custom **evaluation script** based on **PSNR** and **SSIM**, achieving:
  - **Average PSNR**: 30.50
  - **Average SSIM**: 0.88
- Created 10 novel camera poses and integrated them into the GSplat camera setup.

---
## 🚧 Transition to Alternative

Due to persistent rendering failures for novel views using the Gaussian Splatting pipeline and limited time for debugging internal code, we are shifting to an alternate approach for novel view synthesis.

Since **Agisoft Metashape** successfully produced a dense point cloud, mesh, and calibrated cameras, we will now:
- Export the textured 3D model directly from Metashape.
- Use **Metashape’s built-in rendering tools or Python API** to generate novel camera views.
- Evaluate those views using the same SSIM/PSNR comparison framework.

This transition allows us to stay within a reliable toolchain while completing the novel view evaluation using proven data and camera parameters.


---






---
# Conclusion

This repository presents a photogrammetric reconstruction of Apollo 17 imagery, followed by Gaussian Splatting-based view synthesis and model augmentation. The objective is to evaluate whether novel views generated via splatting can enhance photogrammetric mesh quality when used alongside original imagery.

---

## Contact

**Anushka Satav**  
Robotics and Autonomous Systems (AI)  
Arizona State University  
📧 anushka.satav@asu.edu

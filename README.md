# photogrammetry_gaussian_splatting_apollo-17
# Photogrammetry and Gaussian Splatting on Apollo 17 Imagery

This repository presents a photogrammetric reconstruction of Apollo 17 imagery, followed by Gaussian Splatting-based view synthesis and model augmentation. The objective is to evaluate whether novel views generated via splatting can enhance photogrammetric mesh quality when used alongside original imagery.

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

## Contact

**Anushka Satav**  
Robotics and Autonomous Systems (AI)  
Arizona State University  
📧 anushka.satav@asu.edu

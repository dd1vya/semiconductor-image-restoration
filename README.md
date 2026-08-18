# Semiconductor-image-restoration

A NAFNet-based image restoration network enhanced with 2× PixelShuffle super-resolution to remove degradation and recover fine image details.

## Problem Statement

Semiconductor inspection images can suffer from speckle noise, Gaussian degradation, and spatial resolution loss, obscuring fine defect structures.These degradations can occur in different combinations and orders, reducing image quality and important visual details.

The goal is to reconstruct clean, full-resolution images while preserving critical fine details and avoiding artificial artifacts.

The solution must generalize to both in-distribution and unseen image sources.It must also provide fast, efficient inference suitable for GPU-based inspection pipelines.

<img width="799" height="278" alt="image" src="https://github.com/user-attachments/assets/ed9b6c91-13b4-4df2-9c55-f005fcebb7e2" />


## Proposed Architecture

The system combines image restoration and super-resolution into a single end-to-end pipeline.

                 Degraded / NoisyLR Image
                           │
                           ▼
                   Input Convolution
                           │
                           ▼
                    NAFNet Encoder
                           │
                           ▼
                       NAFBlocks
                           │
                           ▼
                  Feature Downsampling
                           │
                           ▼
                  Bottleneck / Middle
                      NAFBlocks
                           │
                           ▼
                  NAFNet Decoder
                           │
                    Skip Connections
                           │
                           ▼
                    2× PixelShuffle
                           │
                           ▼
               Global Residual Learning
                           │
                           ▼
                  Output Reconstruction
                           │
                           ▼
             Restored High-Resolution Image
             
### Global Residual Learning

Instead of reconstructing the complete high-resolution image directly, the network learns a restoration correction over a bicubic-upsampled version of the input.

Conceptually:
```
Input Image
     │
     ├──────────► Bicubic 2× Upsampling ──────┐
     │                                        │
     ▼                                        ▼
NAFNetSR Network ─────► Restoration Residual ─┤
                                              ▼
                                      Final Restored Image
```
This allows the network to focus on recovering missing information and correcting degradation rather than relearning the entire image structure.

## Model Architecture

The proposed model, NAFNetSR


<img width="659" height="590" alt="image" src="https://github.com/user-attachments/assets/bedc3bc7-820c-4e09-8da0-62fb91a7ec95" />



## Training Strategy

The model is trained using paired images:
```

NoisyLR Image  ─────────►  Model  ─────────►  Restored Image
                                                   │
                                                   ▼
                                           Ground Truth Image
```
The training objective is to minimize the difference between the restored output and the ground-truth image.

### Loss Function

A composite loss is used to balance pixel-level accuracy, structural similarity and perceptual quality.

#### 1. L1 Loss

L1 loss measures the pixel-wise difference between the predicted image and the ground truth.

L1 = |Restored Image - Ground Truth|

It helps the model maintain accurate pixel reconstruction and reduces large restoration errors.

#### 2. SSIM Loss

Structural Similarity is used to encourage preservation of image structures such as:

Edges
Patterns
Fine semiconductor structures
Local contrast

#### 3. LPIPS Loss

LPIPS is incorporated to measure perceptual similarity between the restored image and ground truth.

It helps improve the visual quality of reconstructed structures while complementing pixel-level losses.

Composite Objective
Total Loss =
    α × L1 Loss
  + β × SSIM Loss
  + γ × LPIPS Loss

The loss components are combined to balance:

Pixel Fidelity + Structural Similarity + Perceptual Quality

## Dataset

The project uses paired semiconductor inspection images.

#### Ground Truth

Ground-truth images represent clean, full-resolution inspection images.

Supported target resolutions include approximately:

256 × 256
512 × 512

#### Degraded Input

The degraded images contain combinations of:

#### Noise
Reduced spatial resolution
Fine-detail loss

Typical degraded dimensions are approximately:

128 × 128
256 × 256

The model reconstructs the corresponding full-resolution output.


## Training

The training pipeline can be reproduced using:

python train.py

The training script performs the following steps:
```

Load Dataset
     │
     ▼
Preprocess Images
     │
     ▼
Create Training Batches
     │
     ▼
Forward Pass
     │
     ▼
Calculate L1 + SSIM + LPIPS Loss
     │
     ▼
Backpropagation
     │
     ▼
Optimizer Update
     │
     ▼
Validation
     │
     ▼
Save Best Checkpoint
```

Training configuration such as batch size, learning rate, number of epochs and checkpoint path should be specified in the training script.

## Results

Our model compare different loss configurations to determine the most suitable restoration objective.

| Configuration | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| L1 + SSIM + LPIPS (0.6/0.2/0.2) | 27.917 | 0.7655 | 0.1360 |

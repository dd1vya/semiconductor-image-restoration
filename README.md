# semiconductor-image-restoration
An AI-based image restoration and super-resolution system designed to recover clean, high-resolution semiconductor inspection images from degraded, noisy, and low-resolution inputs.

The proposed system uses a NAFNet-based restoration architecture enhanced with 2× PixelShuffle super-resolution and global residual learning. A composite L1 + SSIM + LPIPS loss is used to balance pixel-level accuracy, structural preservation, and perceptual quality.

## Project overview
Semiconductor inspection images can suffer from noise, speckle degradation, reduced spatial resolution, and loss of fine structural details during image acquisition and processing. These degradations can make small features and defects difficult to inspect accurately.
The objective of this project is to restore degraded inspection images while:

Preserving genuine image structures

Recovering fine details

Improving spatial resolution

Reducing noise and degradation

Avoiding unnecessary artificial details

Maintaining good computational efficiency

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

The proposed model, NAFNetSR, contains the following major components:

Input convolution

NAFNet encoder

NAFBlocks

Progressive feature downsampling

Bottleneck / middle NAFBlocks

Decoder with skip connections

2× PixelShuffle upsampling

Global residual reconstruction

Output reconstruction layer

NAFBlocks

NAFBlocks are used for efficient feature extraction and refinement.

The blocks incorporate operations such as:

Normalization

Depthwise convolution

SimpleGate activation

Simplified channel attention

Residual connections

## Dataset

The model is trained using paired degraded and ground-truth images.

Dataset Component	Description
NoisyLR	Degraded low-resolution input image
GT	Ground-truth high-resolution image
Dataset Split
Split	Number of Images
Training	2,880
Validation	320
Testing	400
Total	3,600
```
Input / Output
NoisyLR Image
     │
     ▼
   NAFNetSR
     │
     ▼
Restored Image

128 × 128  ──►  2× SR  ──►  256 × 256
```
The model operates on single-channel grayscale inspection images.

### Data Augmentation

The training pipeline applies lightweight spatial augmentation to improve generalization.

Augmentations

Horizontal flipping

Random 90° rotations

These transformations increase the effective diversity of the training samples while preserving the underlying image structures.

### Loss Function

The final model uses a composite restoration loss:

Total Loss =
    0.6 × L1 Loss
  + 0.2 × SSIM Loss
  + 0.2 × LPIPS Loss
    
#### Composite Loss

Each loss captures a different aspect of image quality:

L1 encourages accurate pixel reconstruction.
SSIM helps preserve structural information.
LPIPS encourages perceptually similar results.

Combining them provides a balance between pixel fidelity, structural preservation, and perceptual quality.

## Training Configuration

Parameter	Value
Framework	PyTorch
Architecture	NAFNetSR
Optimizer	Adam
Epochs	40
Batch Size	16
Scheduler	Cosine Annealing
Loss	L1 + SSIM + LPIPS
Super-Resolution Scale	2×
Device	CUDA GPU / CPU fallback
Checkpoint Selection	Best Validation PSNR

The best-performing checkpoint is selected according to validation PSNR during training.

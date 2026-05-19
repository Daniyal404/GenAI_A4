# Diffusion Models for High-Resolution Image Generation & Reconstruction

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch using **PyTorch** for:

* High-resolution image generation
* Image reconstruction from noisy samples
* Progressive denoising visualization
* Interactive image generation with Gradio

The notebook is fully adapted for **Kaggle GPU environments** and trains on the **CelebA-HQ 256×256** dataset.

---

# Features

* Custom DDPM implementation from scratch
* UNet-based denoising architecture
* Sinusoidal positional embeddings
* Mixed precision training using AMP
* Cosine learning rate scheduler
* Multi-step diffusion sampling visualization
* PSNR and SSIM evaluation metrics
* Gradio interface for interactive image generation
* Kaggle-ready training pipeline

---

# Dataset Structure

The notebook expects the dataset in the following structure:

```text
/kaggle/input/<dataset-name>/celebahq256_imgs/
    train/
    valid/
```

Example:

```text
/kaggle/input/celebahq256/celebahq256_imgs/
    train/
    valid/
```

Both folders should contain image files such as:

* `.jpg`
* `.png`
* `.jpeg`
* `.webp`

---

# How to Run on Kaggle

## Step 1 — Create a New Kaggle Notebook

1. Open Kaggle
2. Click **Code → New Notebook**
3. Enable **GPU**:

   * Notebook Settings → Accelerator → GPU

Recommended GPU:

* Tesla T4
* P100
* V100

---

## Step 2 — Upload or Add Dataset

Add the CelebA-HQ dataset using Kaggle's **Add Data** panel.

The notebook automatically searches inside:

```python
/kaggle/input/
```

No need to upload `kaggle.json` or manually download datasets.

---

## Step 3 — Upload the Notebook

Upload the notebook file:

```text
genai-a4.ipynb
```

Then open it in Kaggle.

---

## Step 4 — Install Dependencies

The notebook already installs required dependencies:

```python
!pip install -q gradio scikit-image
```

Main libraries used:

* PyTorch
* torchvision
* gradio
* scikit-image
* matplotlib
* tqdm
* numpy
* PIL

---

## Step 5 — Configure Training

The central configuration section contains all training parameters:

```python
class Config:
```

You can modify:

* Image size
* Batch size
* Learning rate
* Number of diffusion steps
* Epoch count
* Sampling settings

---

## Step 6 — Start Training

Run all notebook cells sequentially.

The notebook will:

1. Load the dataset
2. Build the diffusion schedule
3. Initialize the UNet model
4. Train the DDPM model
5. Save checkpoints
6. Generate sample images
7. Compute evaluation metrics

---

## Step 7 — Generate Images

After training, the notebook generates images from pure Gaussian noise.

Generated outputs include:

* Final generated images
* Intermediate denoising steps
* Reconstruction comparisons
* PSNR and SSIM scores

---

## Step 8 — Launch Gradio Interface

The notebook includes a Gradio app for interactive image generation.

Features:

* Select number of images
* Control diffusion visualization steps
* Set random seed
* Generate outputs interactively

# Model Architecture

The project uses:

* UNet backbone
* Residual convolution blocks
* Attention mechanisms
* Sinusoidal timestep embeddings
* Gaussian forward diffusion
* Reverse denoising sampling

---

# Evaluation Metrics

The notebook evaluates generated images using:

## PSNR (Peak Signal-to-Noise Ratio)

Measures reconstruction quality.

## SSIM (Structural Similarity Index)

Measures perceptual similarity between generated and original images.

---

# Output Examples

The notebook produces:

* Generated celebrity faces
* Progressive denoising visualizations
* Reconstruction comparisons
* Metric summaries
* Interactive Gradio outputs

---

# Technologies Used

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* Gradio
* scikit-image
* Kaggle Notebooks

---

# Future Improvements

Potential enhancements include:

* DDIM sampling
* Classifier-free guidance
* Attention UNet upgrades
* Faster schedulers
* EMA model weights
* FID score evaluation
* Multi-GPU training
* Latent diffusion support

---

# Conclusion

This project demonstrates a complete DDPM pipeline for high-resolution image generation and reconstruction using PyTorch.

The implementation is:

* Educational
* Modular
* Kaggle-ready
* GPU-optimized
* Compatible with modern PyTorch versions

It provides a strong foundation for understanding and extending diffusion-based generative AI systems.

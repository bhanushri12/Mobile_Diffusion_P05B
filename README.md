# Mobile_Diffusion_P05B
Implementing the "MobileDiffusion: Subsecond Text-to-Image Generation on Mobile Devices"
Certainly! Here’s a detailed README file for your GitHub repository, providing a clear and structured description of your project progression, along with relevant links:

---

# Mobile Diffusion Model: From Basics to Optimized Implementation

Welcome to the repository for the Mobile Diffusion Model. This project encompasses the journey from implementing foundational models to optimizing state-of-the-art diffusion models for efficient deployment on mobile devices.

## Table of Contents
1. [UNet Paper Implementation](#unet-paper-implementation)
2. [Transformer Implementation](#transformer-implementation)
3. [Variational Autoencoder (VAE) with CelebA](#variational-autoencoder-vae-with-celeba)
4. [Optimizing Stable Diffusion Code](#optimizing-stable-diffusion-code)

---

## UNet Paper Implementation

We began with the basic implementation of the UNet model, a pivotal architecture for image segmentation tasks. The UNet model is known for its simple yet powerful encoder-decoder structure, making it an excellent starting point for our diffusion model.

**Steps:**
1. Implemented the basic UNet architecture from scratch.
2. Trained the model on a sample dataset to ensure correctness and understand its workings.

**Reference:**
- [UNet: Convolutional Networks for Biomedical Image Segmentation]([https://arxiv.org/abs/1505.04597](https://arxiv.org/pdf/1505.04597))

- ![WhatsApp Image 2024-05-19 at 22 56 15](https://github.com/bhanushri12/Mobile_Diffusion_P05B/assets/161404554/7172aefd-3d12-4398-9983-aadc7bf15169)


---

## Transformer Implementation

Next, we moved on to implementing the Transformer model from scratch. Transformers are renowned for their efficiency and scalability, particularly in handling sequential data, making them suitable for our diffusion model's needs.

**Steps:**
1. Implemented the Transformer architecture, focusing on the self-attention mechanism translating english to Kannada language.
2. Tested the model with sample data to verify its performance and behavior.

**Reference:**
- [Attention is All You Need]([https://arxiv.org/abs/1706.03762](https://arxiv.org/pdf/1706.03762))

 ![WhatsApp Image 2024-05-19 at 22 56 45](https://github.com/bhanushri12/Mobile_Diffusion_P05B/assets/161404554/df2084b6-f9b6-40e9-844e-c3571ee1e297)


---

## Variational Autoencoder (VAE) with CelebA

Following the Transformer implementation, we proceeded with the Variational Autoencoder (VAE). VAEs are crucial for generative tasks and were trained on the CelebA dataset to validate their effectiveness in generating realistic images.

**Steps:**
1. Built the VAE architecture and implemented the training process.
2. Trained the VAE on the CelebA dataset to generate face images.

**Dataset:**
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

**Reference:**
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

  The generated Images form Decoder are as follows:
  ![vae_1](https://github.com/bhanushri12/Mobile_Diffusion_P05B/assets/161404554/f30bee11-4f6f-4da8-b032-1423393c74c7)
  


---

## Optimizing Stable Diffusion Code

Finally, we optimized the stable diffusion model to make it suitable for mobile deployment. This involved several key optimizations to reduce computational load and parameter count without compromising performance.

**Steps:**
1. Integrated transformer blocks at the UNet bottleneck to leverage low-resolution efficiency.
2. Used cross-attention in pre-bottleneck blocks to enhance computational efficiency.
3. Shared key-value projections in self-attention layers to reduce parameter count.
4. Replaced GELU activation with Swish and fine-tuned softmax into ReLU for mobile optimization.
5. Trimmed feed-forward layers and used separable convolutions in most UNet layers.
6. Pruned redundant residual blocks and employed model distillation techniques.
7. Adopted UFOGen hybrid techniques for fine-tuning with a dual objective.

**Optimized Stable Diffusion Code:**
- [Stable Diffusion Code on GitHub]([https://github.com/CompVis/stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/diffusion.py))

![WhatsApp Image 2024-05-19 at 22 57 08](https://github.com/bhanushri12/Mobile_Diffusion_P05B/assets/161404554/9d439701-64c0-4df9-9f26-532b25e916f4)




## Conclusion

This repository documents our step-by-step progression from basic model implementations to advanced optimizations for mobile diffusion models. Each stage builds upon the previous, culminating in a highly efficient model ready for real-world mobile applications.


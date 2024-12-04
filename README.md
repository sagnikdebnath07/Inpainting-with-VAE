# Inpainting-with-VAE

## Overview
In this project, I implemented a custom **Variational AutoEncoder (VAE)** from scratch and performed image inpainting on the publicly available **CelebA dataset**.  
Due to limited computational resources, I was unable to train a deeper neural network for the encoder and decoder, which could have resulted in more accurate and high-quality image reconstructions.

---

## Dataset
The **CelebA (Celebrities Attributes)** dataset is a large-scale face attributes dataset widely used in computer vision and machine learning research.  
### Key Characteristics:
- Over **200,000 aligned and cropped facial images**.
- Features **40 binary attributes** for each image (e.g., gender, hair color, wearing glasses).
- Popular for tasks like:
  - Face attribute prediction
  - Image generation
  - Facial recognition studies
- Images sourced from celebrities and public figures.
- Developed by researchers at **The Chinese University of Hong Kong**.

For this project:
- Each image was scaled to **256×256×3**.
- Pixel values were **normalized in the range [-1, 1]**.

---

## Variational AutoEncoder (VAE)
### Encoder
- **Architecture**:  
  - **Conv2d layers** with:
    - Kernel size: `6`
    - Stride: `4`
    - Padding: `1`
  - Image dimensions reduce by a factor of 4 after each layer.
  - 3 layers in total, resulting in a final feature size of **4×4×64**.
- **Activation function**:  
  - **SiLU** (Swish activation function).  
- **Latent dimensions**:  
  - Set to `128`.

### Decoder
- **Architecture**:  
  - Mirrored the encoder structure to reconstruct the input dimensions of **256×256×3**.
- **Final activation function**:  
  - **Tanh**, to match the pixel range of the images.

### Loss Function

The VAE loss function is the sum of two components:  
1. **Reconstruction Loss**: Measures how well the decoder reconstructs the input.  
2. **KL Divergence (KLD)**: Ensures the latent space follows a standard normal distribution.  

The total loss is given by:

```math
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \mathcal{L}_{\text{KL}}\
```

#### 1. Reconstruction Loss
For continuous data (e.g., images), the reconstruction loss is often the Mean Squared Error (MSE) or Binary Cross-Entropy (BCE):

\[
\mathcal{L}_{\text{reconstruction}} = \mathbb{E}_{q(z|x)} \left[ \| x - \hat{x} \|^2 \right]
\]

Where:  
- \(x\): Original input.  
- \(\hat{x}\): Reconstructed input.  
- \(q(z|x)\): Latent distribution.

#### 2. KL Divergence
The KL Divergence term ensures that the learned latent distribution \(q(z|x)\) is close to the prior \(p(z)\), typically a standard normal distribution \(\mathcal{N}(0, I)\):

\[
\mathcal{L}_{\text{KL}} = D_{\text{KL}} \left( q(z|x) \| p(z) \right)
\]

For Gaussian latent variables, this can be computed as:

\[
\mathcal{L}_{\text{KL}} = \frac{1}{2} \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)
\]

Where:  
- \(\mu\): Mean of the latent distribution.  
- \(\sigma^2\): Variance of the latent distribution.

#### Final Loss
The combined loss becomes:

\[
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \frac{1}{2} \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)
\]


---

## Training
- **Batch size**: `8`.
- **Optimizer**:  
  - **Adam**, with an initial learning rate of `3e-4`.
- **Epochs**:  
  - Trained for `10 epochs` (preferable training duration is 50-100 epochs for better results).  
  - Limited training due to computational constraints.

---

## Inference
The **image inpainting results** can be found in the accompanying notebook.  
While the reconstruction quality is not very high, the model demonstrates the ability to perform inpainting where required.

---

## Challenges and Future Improvements
### Challenges:
1. **Lack of computational resources**:
   - Unable to train deeper models that could better understand patterns.
2. **Limited training duration**:
   - Model trained for only 10 epochs, affecting reconstruction quality.

### Future Improvements:
1. Train the model for more epochs (50-100 epochs) to achieve better reconstructions.
2. Use deeper encoder and decoder architectures to improve the model's ability to learn and generalize.
3. Explore advanced techniques like **attention mechanisms** or **GAN-based approaches** for enhancing image inpainting quality.

---

## Acknowledgments
- **CelebA Dataset**: Developed by researchers at **The Chinese University of Hong Kong**.
- Special thanks to the open-source machine learning and deep learning community for making this project possible.

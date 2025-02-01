# Inpainting-with-VAE

## Overview
In this project, I implemented a custom **Variational AutoEncoder (VAE)** from scratch and performed image inpainting on the publicly available **CelebA dataset**.  
Due to limited computational resources, I was unable to train the model for longer epochs, which could have resulted in more accurate and high-quality image reconstructions.

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
    
    $\text{Swish}(x) = x \cdot \sigma(x)$
    
    Where:
    - $x$ is the input.
    - $\sigma(x)$ is the **Sigmoid function**
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
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \mathcal{L}_{\text{KL}}
```

#### 1. Reconstruction Loss
For continuous data (e.g., images), the reconstruction loss is often the Mean Squared Error (MSE) or Binary Cross-Entropy (BCE):

```math
\mathcal{L}_{\text{reconstruction}} = \mathbb{E}_{q(z|x)} \left[ \| x - \hat{x} \|^2 \right]
```

Where:  
- $x$: Original input.  
- $\hat{x}$: Reconstructed input.  
- $q(z|x)$: Latent distribution.

#### 2. KL Divergence
The KL Divergence term ensures that the learned latent distribution $q(z|x)$ is close to the prior $p(z)$, typically a standard normal distribution $\{N}(0, I)$:

```math
\mathcal{L}_{\text{KL}} = D_{\text{KL}} \left( q(z|x) \| p(z) \right)
```

For Gaussian latent variables, this can be computed as:

```math
\mathcal{L}_{\text{KL}} = \frac{1}{2} \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)
```

Where:  
- $\mu$: Mean of the latent distribution.  
- $\sigma^2$: Variance of the latent distribution.

#### Final Loss
The combined loss becomes:

```math
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \frac{1}{2} \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)
```


---

## Training
- **Batch size**: `8`.
- **Optimizer**:  
  - **Adam**, with an initial learning rate of `3e-4`.
- **Epochs**:  
  - VAE from Scratch notebook has model trained for `10 epochs` (preferable training duration is 50-100 epochs for better results).
  - VAE 2 has model trained for `30 epochs`
  - Limited training due to computational constraints.

---

## Inference
- The **image inpainting results** for the model trained for 10 epochs can be found in the Image Inpainting notebook.  
- The **image inpainting results** for the model trained for 30 epochs can be found in the VAE 2 notebook.  
- While the reconstruction quality is not very high, the model demonstrates the ability to perform inpainting where required.
- Inferences for model trained for 10 epochs:
  - <img src="10 epoch pic 1.jpg" alt="GitHub Logo" width="400">
  - <img src="10 epoch pic 2.jpg" alt="GitHub Logo" width="400">
- Inferences for model trained for 10 epochs:
  - <img src="30 epochs pic 1.jpg" alt="GitHub Logo" width="400">
  - <img src="30 epochs pic 2.jpg" alt="GitHub Logo" width="400">


---

## Challenges and Future Improvements
### Challenges:
1. **Lack of computational resources**:
   - Unable to train deeper models that could better understand patterns.
2. **Limited training duration**:
   - Model trained for only 10 and 30 epochs, affecting reconstruction quality.
   - Model trained for 30 epochs shows significant reconstruction quality over model trained for 10 epochs.

### Future Improvements:
1. Train the model for more epochs (50-100 epochs) to achieve better reconstructions.
2. Use deeper encoder and decoder architectures to improve the model's ability to learn and generalize.
3. Explore advanced techniques like **attention mechanisms** or **GAN-based approaches** for enhancing image inpainting quality.

---

## Acknowledgments
- **CelebA Dataset**: Developed by researchers at **The Chinese University of Hong Kong**. [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [Blog](https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/) by Hunter Heidenreich.
- [Blog](https://lilianweng.github.io/posts/2018-08-12-vae/) by Lilian Weng
 # Inpainting-with-VAE
This is the implementation of the paper "Image Inpainting with Deep Generative Models"


def fibonacci_sequence(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        list_fib = [0, 1]
        while len(list_fib) < n:
            next_fib = list_fib[-1] + list_fib[-2]
            list_fib.append(next_fib)
        return list_fib # Inpainting-with-VAE
This is the implementation of the paper "Image Inpainting with Deep Generative Models"


def fibonacci_sequence(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        list_fib = [0, 1]
        while len(list_fib) < n:
            next_fib = list_fib[-1] + list_fib[-2]
            list_fib.append(next_fib)
        return list_fib
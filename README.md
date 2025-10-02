# Generative Models: Practical Notebooks 

This repository contains my lab works for the Generative Models course. Each lab explores a key generative approach, combining theoretical insights with hands-on PyTorch implementation. Below is a summary of the content and skills applied in each lab.

## Lab 1: Convolutional GAN on MNIST
- **Objective**: Implement and train a Generative Adversarial Network (GAN) with convolutional architectures on the MNIST dataset.
- **Key Topics**:
  - Architecture of DCGANs (Deep Convolutional GANs).
  - Training dynamics between generator and discriminator.
  - Latent space exploration through interpolation between generated digits.
- **Skills Applied**:
  - Implementation of WGANs training with Gradient Penalty regularization.
  - Generation and visualization of interpolated samples.
  - Evaluation of generative quality using nearest neighbor comparisons.
 
## Lab 2: WGAN on a 2D Synthetic Dataset
- **Objective**: Learn and analyze Wasserstein GANs (WGAN and WGAN-GP) on a simple 2D dataset, and compare their behavior with a standard GAN.
- **Key Topics**:
  - Training the discriminator alone (fixed generator) with weight clipping and estimation of its Lipschitz constant.
  - Training the discriminator alone (fixed generator) with gradient penalty (WGAN-GP).
  - Simultaneous training of WGAN-GP (generator and discriminator) and study of hyperparameter impact on stability.
  - Analysis of generator training when the discriminator is fixed.
  - Comparison with a standard GAN: discriminator-only training and joint training.
- **Skills Applied**:
  - Implementation of WGAN and WGAN-GP training procedures in PyTorch.
  - Implementation of weight clipping and gradient penalty to enforce the Lipschitz constraint.

## Lab 3: Variational Autoencoder (VAE) and Generative Experiments
- **Objective**: Explore autoencoders and variational autoencoders for image reconstruction and generation, and compare different strategies to sample from the latent space.
- **Key Topics**:
  - Evaluation and comparison of a vanilla autoencoder / VAE : qualitative and quantitative assessment of reconstruction quality and generalization to the test set.
  - Two sampling/generation methods from the autoencoder latent space by assuming a Gaussian latent distribution estimated from training statistics (mean and covariance), with two covariance models:
    - diagonal covariance (independent latent dimensions),
    - full covariance matrix (capturing latent correlations).
  - Comparison of generated images across methods both qualitatively and with an Inception Score computed using a convolutional classifier.
- **Skills Applied**:
  - Implementation of vanilla autoencoder and VAE architectures in PyTorch.
  - Empirical evaluation of reconstruction quality and generalization on held-out data.
  - Estimation of latent Gaussian statistics from training encodings (mean, variance, diagonal vs full covariance) and sampling from those models.
  - Training a β-VAE (implementation of the β loss) and comparing its behavior to non-probabilistic AEs.
  - Training a convolutional classifier to compute an Inception-like score, and using that metric alongside visual inspection to compare generation quality across methods.



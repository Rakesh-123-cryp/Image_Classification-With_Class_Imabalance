# Fasion_Image_Classification-With_Class_Imabalance

## What is Class Imabalance

When the number of samples available for each class is not equal it is said to be a class imbalance problem. The imbalance can vary depending on the ratio of the data count between the classes.
The figure given below shows a visual representation of the problem.
# ![Imbalance Problem Image](/images/imbalance.)

## Different Solutions available

### 1. Data Augmentation

This is a very common and effective method that can be exploited for the imbalance problem but augmenting the data more than the ratio of the classes can negatively affect the model's performance.

There are two kinds of data augmentation:
* **Augmenting using random rotations and transformations**
  This is considered when the data is very scarce and not enough to generate examples. There are two different knids of augmentation. The **_Spatial Augmentation_** and **_Pixelwise Augmentation_**
  ### ![Imbalance Problem Image](/images/augmentation.)
  
  - The Spatial Augmentation performs rotation, transformation or crops to alter the spatial details in the image.
  - The Pixelwise Augmentation changes the luminance, intensity and contrast of the image
  It is safer to not have each augmentation technique double the size of the data images.

* **Augmenting using genrative models such as VAE or VQ-VAE**
  - VAEs are models that can learn the data distribution assuming the latent distribution learned is one of Gaussian.
  - The true posterior distribution (The true distribution of the image pixels) are intractable, so VAE network uses an encoder to approximate the posterior into a gaussiam distribution.
  - The encoder encodes the input image into the Gaussian distribution whose mean and variance is predicted after which the decoder samples from this latent distribution using the reparameterization trick to generate an image.
 

# AgeGenderNet
Age and gender prediction from facial images using a multitask ResNet-18 model.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parhamahn/AgeGenderNet/blob/main/train.ipynb)

# ResNet18: Age-Gender Estimation Model

## Why Age and Gender Prediction Matters
Ever wondered how apps can guess your age or tailor content to you? That’s where age and gender prediction comes in! It’s used in:

- **Personalization:** Ads and content that actually match your demographic.  
- **Security:** Monitoring or authentication systems that need to know who’s who.  
- **Healthcare:** Studying age-related trends or spotting conditions early.  
- **Being bored:** If you’re someone like me who gets bored, why not dive into a fun project like this?

By teaching AI to estimate age and gender from images, we can make smarter, more human-aware systems.

## Dataset
For this project, we are using the **UTKFace** dataset. It’s a massive collection of faces from ages 0 to 116, along with gender labels. The dataset also includes ethnicity and timestamps, but we’ll stick to age and gender for simplicity.

## Image Preprocessing
Before feeding images into our model, we do some basic prep:

- **Resize:** Make sure all images are the same size.  
- **Normalize:** Scale pixel values so the model learns faster and better.  
- **Data Augmentation (optional):** Flip, rotate, or crop images to make the model more robust.

These steps ensure our model sees clean, consistent images and can actually learn meaningful features from them.

## Model
We’re using **ResNet18**, a type of CNN (Convolutional Neural Network) introduced by Microsoft in 2015. The “18” refers to the number of layers with trainable weights.

Deep networks can run into **vanishing gradients**—gradients shrink as they move backward through layers, making learning slow or impossible. Residual connections in ResNet fix this, letting gradients flow smoothly and helping the network actually learn.

Think of it like this: the network wants to pass information backward through many layers, but without shortcuts, it gets tired and loses signal. Residual connections give it a “fast lane” to keep the learning alive.

![test_image_output1](https://github.com/user-attachments/assets/1ad1e647-f351-4b77-9c24-e521604de98e)

## How It Works (Brief)
1. The model takes a facial image as input.  
2. Convolutional layers extract features like eyes, nose, and mouth patterns.  
3. Residual connections make sure these features are propagated effectively.  
4. Fully connected layers finally predict **age** (as a number) and **gender** (as male/female).  

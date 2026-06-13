# Team Diary Guidelines

Write here information that you might think is relevant for your colleagues to know about. Do so as you personally prefer; bullet points are fine, summarized versions are fine, no pressure. You may leave spots empty as you please. This is meant for us, the bar is low. You may edit the format if you feel like it needs improvement 

Format:
``` 
## DD-MM-YYYY

### Worked On

### Learned

### Problems

### Next

## DD-MM-YYYY

### Worked On
.....
```

### Antti

## 04-06-2026

### Worked on
CNN classifier for different graph types

### Learned

After some background research, decided to start working with PyTorch. Aapo found a large dataset on Kaggle that contains hundreds of thousands labeled graph images of several different types. The provided metadata also includes the actual data values on the graphs but for the CNN classifier these are not needed. Defining networks using PyTorch seems simple enough and different network architectures should be easy to realise since PyTorch offers a wide variety of different kinds of layers that can be used sequentally.

### Problems

After doing looking at the dataset, it is immediately obvious that the image resolutions are not consistent, which is a problem for a CNN. A simple solution would  be to pad each image to a fixed resolution. I will need to write a script that checks the resolution of each image and finds the maximum in both directions before training.

### Next

Check the resolution distributions and start working on CNN training

### Antti

## 05-06-2026

### Worked on
CNN classifier for different graph types

### Learned

CNN training requires a specific file structure for organising the images. Found a library (split-folders) to order the images automatically into training/validation/testing sets randomly with specific proportions. A nice feature is that a seed value can be used when sampling the images, which should make the network training more reproducible with the same dataset. 

Also studied some basic PyTorch functions to transform the input data. Instead of e.g. padding, the images can simply be just resized to a given resolution using various interpolation schemes. This could be used to also reduce the image size, which would also mean smaller networks. 

### Problems

With very little prior experience, choosing the correct data transformations and training options seems like a pretty daunting task. A lot of the tutorials use datasets that come with the library instead of using custom datasets.

### Next

Implement and train a simple CNN. At first, the plan is to only classify horizontal and vertical bar graphs.


### Antti

## 04-06-2026

### Worked on
CNN classifier for different graph types

### Learned

After some background research, decided to start working with PyTorch. Aapo found a large dataset on Kaggle that contains hundreds of thousands labeled graph images of several different types. The provided metadata also includes the actual data values on the graphs but for the CNN classifier these are not needed. Defining networks using PyTorch seems simple enough and different network architectures should be easy to realise since PyTorch offers a wide variety of different kinds of layers that can be used sequentally.

### Problems

After doing looking at the dataset, it is immediately obvious that the image resolutions are not consistent, which is a problem for a CNN. A simple solution would  be to pad each image to a fixed resolution. I will need to write a script that checks the resolution of each image and finds the maximum in both directions before training.

### Next

Check the resolution distributions and start working on CNN training

### Antti

## 11-06-2026

### Worked on
CNN classifier

### Learned

Managed to understand PyTorch documentation enough to make a template for basic CNN architecture. Also added one-hot encoding to the training data to make the target labels work with error functions and wrote a basic version of the training loop.

### Problems

Actually tarining the network is computationally too demanding for my own machine. Also had some issues with setting up repo on my own end, resulting in not being able to push changes to the remote repo. Fixed by cloning the repo again using SSH instead of HTTPS.

### Next

Refactor the CNN trainer and add the possibility to process arbitrary images with the network.

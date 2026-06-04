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

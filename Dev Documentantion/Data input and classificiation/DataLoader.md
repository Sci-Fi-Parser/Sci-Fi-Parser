# Team Diary Guidelines

Write here information that you might think is relevant for your colleagues to know about. Do so as you personally prefer; bullet points are fine, summarized versions are fine, no pressure. You may leave spots empty as you please. This is meant for us, the bar is low. You may edit the format if you feel like it needs improvement.

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

## 02-06-2026
### Niko
#### Worked On
Testing out PyMuPDF for getting images from PDFs.

#### Learned
Some PDFs may contain the data for the chart directly as a vector graph. This could be extracted when possible and be used for later steps. 

#### Problems
The vector graphs of the charts may come in pieces. Maybe they could be reconstructed into one graph? Or even just give partial context. I'm also not sure if the points they give tell directly the values for each key.

#### Next
See if PyMuPDF could get fuller images. Otherwise need to checkout other tools.

## 03-06-2026
### Niko
#### Worked On
Getting better images with PyMuPDF.

#### Learned
Vector graphics (or drawings) can be clustered into a single image. In this it loses the exact points and returns just a Rect object. Some PDFs may have the graphs as either images or drawings so both should be extracted. Classification should weed out non-graphs.

#### Problems


#### Next
PDF / folder input and figuring out how it will fit in the rest of the pipeline. Also need to figure out if "exact data" is applicable for data extraction.
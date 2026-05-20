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

## 19-05-2026

### Worked On
OCR = Optical character recognition, CV = Computer vision
Looked for how to use OCR in the pipeline and which tools could be good

### Learned
The data extraction part in the pipeline we can do the OCR in some different ways. 

1. We can put the outputs of the OCR/CV into the VLM 

2. We can run the VLM separately without the information and then compare it with the separate OCR/CV information. 

In the first option we can accidentally poison the VLM if the OCR/CV information is wrong. Need to do testing to figure out some threshold for when to use the OCR/CV info in VLM.

In the second option if the VLM is not numerically good enough without the OCR/CV context clues it doesn't give a good reference point for comparison. 
How you merge the two inputs into one can be hard as well. It could be possibly a smaller LLM or a VLM.

Either way we should do the OCR/CV processing before making the decision.

PaddleOCRv5 seems maybe the strongest OCR option, EasyOCR seems also decent, Tesseract might be a little outdated.

Some kind of image preprocessing needs to be done to the images, different tools might prefer different things.

### Problems

Don't really know how trustworthy or accurate the OCR/CV is

### Next

Start implementing and experimenting with OCR

For the first pipeline we should probably do the option 1. where we just don't pass the information if it is low confidence


## 20-05-2026

### Worked On
Research on how OCR and CV tools interact and which tools could be used.

### Learned
OCR, CV and probably some manually written code make up the pre-VLM part of the pipeline.

OpenCV seems probably the best CV library, it can also be used for preprocessing.

This can be a rough example of the potential pipeline.

1. Preprocess the image
  - Upscale? (To *some* resolution)
  - Increase contrast?
  - Binariwation?
  - CV seems to prefer mostly non-altered images, OCR can benefit more on preprocessing.
  - Store all variants created

2. Detect chart structure
  - Plot area
  - Axes
  - Gridlines
  - Panels
  - Tick marks

3. OCR text regions with PaddleOCRv5
  - Axis values and title
  - legend
  - Bar values (if present)

4. Build coordinate system
  - Linear or log scale or categories
  - y pixel -> value 
  - x pixel -> value, category etc.

5. Extract coordinate data
  - What is the y-coordinate of the top/bottom of bar graph
  - Do this for every x-point
  - Compare with bar values if they exist

- If the bar values exist later steps might not be needed

### Problems

### Next

Exploring OpenCV and PaddleOCRv5 at code level.

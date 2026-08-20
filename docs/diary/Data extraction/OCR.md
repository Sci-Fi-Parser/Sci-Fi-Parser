## 19-05-2026

### Aapo

#### Worked On
OCR = Optical character recognition, CV = Computer vision
Looked for how to use OCR in the pipeline and which tools could be good

#### Learned
The data extraction part in the pipeline we can do the OCR in some different ways. 

1. We can put the outputs of the OCR/CV into the VLM 

2. We can run the VLM separately without the information and then compare it with the separate OCR/CV information. 

In the first option we can accidentally poison the VLM if the OCR/CV information is wrong. Need to do testing to figure out some threshold for when to use the OCR/CV info in VLM.

In the second option if the VLM is not numerically good enough without the OCR/CV context clues it doesn't give a good reference point for comparison. 
How you merge the two inputs into one can be hard as well. It could be possibly a smaller LLM or a VLM.

Either way we should do the OCR/CV processing before making the decision.

PaddleOCRv5 seems maybe the strongest OCR option, EasyOCR seems also decent, Tesseract might be a little outdated.

Some kind of image preprocessing needs to be done to the images, different tools might prefer different things.

#### Problems

Don't really know how trustworthy or accurate the OCR/CV is

#### Next

Start implementing and experimenting with OCR

For the first pipeline we should probably do the option 1. where we just don't pass the information if it is low confidence


## 20-05-2026

### Aapo

#### Worked On
Research on how OCR and CV tools interact and which tools could be used.

#### Learned
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

#### Problems

#### Next

Exploring OpenCV and PaddleOCRv5 at code level.

### Niko

#### Worked On
Testing how to get PaddleOCRv5 running and how it works.

#### Learned
PaddleOCR has multiple optional capabilities, some of which seemed promising at first glance. However, information extraction seems to be more for traditional documents rather than graphs. PaddleOCR also has a doc-parser, which may be of note. Whether we will use it or not will probably depend on how lightweight it is. I have not yet looked further into it.

I tested PaddleOCR to see if the results were accurate. In addition I measured how much time it takes to process different graphs after the models and weights had been loaded in.

Comparing the inference engines: paddle_static was much faster than transformer: 7.9 vs 29.4 seconds on the same image. Therefore it is likely the better the option unless integration with the Hugging Face ecosystem is more important.

PaddleOCR fared well with simple bar graphs where figures were unobsructed and aligned normally. It was unable to properly read text rotated 90° or text placed on top of a line in a line graph. PaddleOCR has a text line orientation classification module, though, it requires for the labels to first be split up before it can recognize if the label is rotated. Images may thus require preprossessing.

#### Problems
Rotated text needs to be properly handled. But perhaps this should be tackled only after the OCR is known to work sufficiently in the pipeline on a simple barchart. 

#### Next
Figure out how the data could be used in conjuction with the VLM. Test other OCR unless PaddleOCR is deemed good enough.


## 23-05-2026

### Aapo

#### Worked On
Research and starting implementing CV:
Using OpenCV to detect lines and bar chart bars.
The implementation is on the OcrCV branch

#### Learned
OpenCV operates on images as numpy.ndarray
Same things in OpenCV can be done in many ways
What preprossessing we do matters a lot, might need to do some separate color maps depending on the image.
Line detection should be nice for most graph types, needs a lot more work.
Some automated accuracy test for this part is needed

#### Problems
A lot more refining and experimentation needed for bars and especially lines.
- Lines currently get spotted at most text spaces
- Quite a few bar charts don't have actual x and y-axes.
- Need to merge similar close lines into one.
- Weird colored charts don't work with the binarization techniques. 
- charts with bars next to each other don't work cleanly

#### Next
Bar detection testing and refinement

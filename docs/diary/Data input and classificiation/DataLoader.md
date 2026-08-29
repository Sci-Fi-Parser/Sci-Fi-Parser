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

## 05-06-2026
### Niko
#### Worked On
Getting higher resolution images

#### Learned
The dpi can simply be changed on a pixmap and the downscaled to the desired size. PyMuPDF has methods to work with Pillow.

#### Problems
The CNN will need a consistent size in both directions.

#### Next
Probably check on rotating vertical images. Also need to check how it will fit into the whole pipeline.


## 09-06-2026 - 10-06-2026
### Aapo
#### Worked On
Making the sub-pipeline around loading pdfs and the eventual output that can be used by other parts.
Connected loading and storing to the parser made by Niko and did some tweaks and bug fixes.

#### Learned
Parsing 59 pdfs took: ~70s (writing images to disc), ~50s (no disk writes). This has high variability between individual pdfs.

#### Problems
Some edge cases seem to make parsing some pdfs (TGV 2018 Q4 Shareholder Letter) really slow or just broken.
The TGV 2018 Q4 Shareholder Letter from the presentations folder took more than 30 min on its own before I cancelled it.

Some parsed images are still somewhat malformed, very large or weird in other ways, got some height = 0 related errors etc.

#### Next
Discussion and work on how the writer can store the metadata returned, how this data is used in the classification

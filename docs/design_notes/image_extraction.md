# Image extraction

We first used pymudpdf to extract images from PDFs, but noticed later that it had an AGPL license, so we switched to pypdfium2. It lacks image clustering, so we had to write it ourselves.

The extraction pipeline is given a path to a PDF. The PDF is parsed page-by-page, and from them both embedded images and vector graphics are extracted. Note that vector drawings need to be clustered to form the whole graph. However, scanned images will result as whole pages as images are not cropped.

Image metadata included:
- image_id: hash based on image
- metadata
    - pdf_id
    - page_number
    - source_type: embedded_image or vector_drawing

PDF metadata included:
- pdf_id: hash based on PDF content
- metadata
    - file_name
    - page_count

Images and PDFs are linked through pdf_id.
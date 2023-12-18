# Document Paddle OCR

This class serves as a wrapper for PaddleOCR, providing functionalities to perform Optical Character Recognition (OCR) tasks on images and documents.

## Class: Paddle_OCR

### Initialization

The `Paddle_OCR` class initializes the Paddle OCR model for text recognition.

### Methods

### Method 1: `apply_ocr`
#### `apply_ocr(self, img_path: str) -> List[List[int, int, int, int], List[str, float]]`

This method performs OCR on the given image and retrieves bounding boxes, words/texts, and their associated confidence scores.

- #### Arguments:
    - `image_path`: Path of the image to be OCR [`image` or `pdf`].

- #### Returns:
    - List of list which contains coordinates of the bounding boxes, words, confidance score.
    - `[[[c1, c2, c3, c3], ["ocr text", 0.92435]]]`

#### Example:
```python

from doc_intelligence.model.ocr.paddleocr import Paddle_OCR 

ocr_obj = Paddle_OCR()
results = ocr_obj.apply_ocr("./path/to/image.jpg")
print(results)
```

### Method 2: `draw_multiple_boxes`
#### `draw_multiple_boxes(self, image_path : str, ind_list: List[List[int, int, int, int]]) -> PIL.Image`

The `draw_multiple_boxes` method draws bounding boxes over the detected words in an image.

- #### Arguments:
    - `image_path`: Path of the image.

- #### Returns:
    - PIL Image

#### Example:
```python

from doc_intelligence.model.ocr.paddleocr import Paddle_OCR 

ocr_obj = Paddle_OCR()
resutls = ocr_obj.apply_ocr("./path/to/image.jpg")
bounding_boxes = [ x[0] for x in resutls]

image_with_bboxes = ocr_obj.draw_multiple_boxes("./path/to/image.jpg", bounding_boxes)
image_with_bboxes.show()
```
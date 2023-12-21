
import os
import cv2
import PIL
import copy
import math
import array
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from typing import Any, Dict, List, Tuple
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFont
from typing import List


class Paddle_OCR(PaddleOCR):

    """
    wrapper class for the paddleocr
    """
    def __init__(self, language: str = "en"):

        """
        Args: 
            language (str): content language of your document
        Returns: 
            None
        """
        self.language = language
        self.ocr_model = PaddleOCR(lang=self.language)

    def apply_ocr(self, img_path: str) -> List:

        """
        Perform OCR on the given image and return bounding boxes, words and scores

        Args: 
            img_path (str): path to the image document
        Returns: 
            List of list where each elements contains bounding boxes, words/texts and scores 
            [[[c1, c2, c3, c3], ["ocr text", 0.92435]]]
        
        """
        result = self.ocr_model.ocr(img_path, cls=True)

        try :
       
            flat_result = [item for sublist in result for item in sublist]

            self.boxes = [line[0] for line in flat_result]
            self.texts = [line[1][0] for line in flat_result]
            self.scores = [line[1][1] for line in flat_result]

            

        except:

            flat_result = [item for sublist in result for item in sublist]

            self.boxes = [line[0] for line in result]
            self.texts = [line[1][0] for line in result]
            self.scores = [line[1][1] for line in result]


            flat_result = result 

        return flat_result

    def draw_multiple_boxes(self, image_path : str, ind_list: list) -> PIL.Image:

        """
        Draw bounding boxes over all the detected words

        Args: 
            image_path (str): path to the image
            ind_list (list): list of the bounding box coordinates
        Returns: 
            PIL Image with all the printed bounding box 
        
        """

        image = Image.open(image_path)
        box_img = copy.deepcopy(image)

        for index in ind_list:
            cv2.rectangle(
                box_img,
                [int(x) for x in index[0][0]],
                [int(x) for x in index[0][2]],
                color=[0, 0, 0],
                thickness=1,
            )

        return box_img

    def ocr_img_alt(self, img_path):
        
        result = self.ocr_model.ocr(img_path, det=True, rec=False, cls=True)

        flat_result = [item for sublist in result for item in sublist]
        image = Image.open(img_path).convert("RGB")
        boxes = [line[0] for line in flat_result]
        image_array = Image.open(img_path)

        return image, flat_result, image_array

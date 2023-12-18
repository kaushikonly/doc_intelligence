import paddle
from paddleocr import PaddleOCR,draw_ocr
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import array
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from img2table.document import PDF, Image as Imagedocument
from img2table.ocr import TesseractOCR
from PIL import Image as PILImage
import pandas as pd
import math as mathz
from difflib import SequenceMatcher
from IPython.display import HTML
import math
import re
from collections import Counter
import jellyfish
from pprint import pprint
from paddlenlp import Taskflow
from sklearn.model_selection import train_test_split 
from doc_intelligence.model.ocr.paddleocr import Paddle_OCR
#from doc_transform.model.ocr.paddleocr import Paddle_OCR
from utils.helper.text_processing import Text_Manipulation
from utils.helper.computations import Boundingbox_Computations
from doc_intelligence.transform.doc_pre_processor import Doc_Preprocessor
from extract.structure import Document_structure
from pandas import DataFrame
import pandas as pd
import yaml

import argparse

paddle.utils.run_check()

class Document_Information():


    def __init__(self):
        pass


    def extract_table(self, fpath : str , path_yaml='/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/test_names.yaml') -> pd.DataFrame:

        """

        Given an input image of a lab report, extract_table will provide a dataframe, containing the medical tests, their values and ranges as output.

        Args:

        fpath: Path to .jpeg of input image of lab report 
        path_yaml: path to yaml file containing medical tests to look for


        Returns:

        dataframe containing the medical tests, their values and ranges


        """


        with open(path_yaml, 'r') as file:
            tests = yaml.safe_load(file)
        test_list = tests['test_name']


        OCR_model = Paddle_OCR()
        Text_manip = Text_Manipulation()
        Bounding_box_comp = Boundingbox_Computations()
        Doc_struct = Document_structure()


        im_bottom, im_top = Doc_struct.crop_with_jaro(fpath, path_yaml)
        im_bottom.save('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/bottom/bottom.jpg')
        im_top.save('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/top/top.jpg')
        bounding_boxes = OCR_model.apply_ocr('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/bottom/bottom.jpg') 
        test_location = Text_manip.select_subset_boundingbox(bounding_boxes,test_list)
        pairwise_bounding_box_relations = Bounding_box_comp.inter_bounding_box_relations(test_location['jaro_matching'],bounding_boxes)


        key_pairs_all = Text_manip.row_extraction_slope_method(pairwise_bounding_box_relations,test_list)


        d = Doc_struct.table_formation(key_pairs_all,path_yaml)

        return d





#obj = Document_Information()
#d = obj.extract_table('/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/labreport.jpeg', '/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/test_names.yaml')
#print(d)


obj = Document_Information()
d = obj.extract_table('/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/labreport.jpeg', '/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/test_names.yaml')
print(d)


#im_bottom.save('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/bottom/bottom.jpg')
#im_top.save('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/top/top.jpg')
#bounding_boxes = OCR_model.apply_ocr('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/bottom/bottom.jpg') 
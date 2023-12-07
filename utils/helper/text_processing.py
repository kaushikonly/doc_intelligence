
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
import argparse

paddle.utils.run_check()






class Text_Manipulation():

    def extract_txt(self, bounding_boxes):
        
        txts = [line[1][0] for line in bounding_boxes]
        
        return txts

    def jaro_comparison(self, txts,test_list):
        
        Jaro_Dist_All = []
        for indiv_txt in txts:   

            Jaro_Dist = []

            for indiv_test in test_list:


                    indiv_txt = indiv_txt.lower()
                    indiv_test = indiv_test.lower()
                    jaro_dist = jellyfish.jaro_distance(indiv_txt, indiv_test)
                    Jaro_Dist.append(jaro_dist)
            Jaro_Dist_All.append(np.max(Jaro_Dist))
            
        return Jaro_Dist_All

        

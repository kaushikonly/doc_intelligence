
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




class boundingbox_computations:

    def euc_dist(self, point1,point2):
        dist = np.linalg.norm(point1 - point2)
        return dist 


    def skew_calculate(self, bounding_box_properties):
        
        angle_all = bounding_box_properties['angle_all']
        box_length_all = bounding_box_properties['box_length_all']
        
        box_length_all_norm = box_length_all/sum(box_length_all)   
        angle_correction = np.dot(angle_all,box_length_all_norm)


        return angle_correction


    def bounding_box_property_estimator(self, bounding_boxes):
    

        bounding_box_properties = {}
        center_coord = []
        boxcontent_center_all = []
        boxcontent_coord_all_x = []
        boxcontent_coord_all_y = []
        box_length_all = []
        text_all = []
        slope_all = []
        angle_all = []

        for i,bla in enumerate(bounding_boxes):  

            box_coord = np.array(bounding_boxes[i][0])
            text_all.append(bounding_boxes[i][1][0])
            box_xcoord = box_coord[0:4,0]
            box_ycoord = box_coord[0:4,1]

            boxcenter_xcoord = np.mean(box_coord[0:4,0])
            boxcenter_ycoord = np.mean(box_coord[0:4,1])
            slope_1 = (box_ycoord[1]-box_ycoord[0])/(box_xcoord[1]-box_xcoord[0])
            slope_2 = (box_ycoord[2]-box_ycoord[3])/(box_xcoord[2]-box_xcoord[3])

            slope_mean = (slope_1+slope_2)/2
            angle = math.degrees(math.atan(slope_mean))
            angle_all.append(angle)
            box_center = array.array('d', [boxcenter_xcoord, boxcenter_ycoord])
            boxcontent_center = [bounding_boxes[i][1][0], box_center]
            boxcontent_center_all.append(boxcontent_center)
            center_coord.append(box_center)
            boxcontent_coord_all_x.append(box_coord[0:4,0])
            boxcontent_coord_all_y.append(box_coord[0:4,1])
            slope_all.append(slope_mean)
            
            p0 = np.array((box_xcoord[0], box_ycoord[0]))
            p1 = np.array((box_xcoord[1], box_ycoord[1]))
            dist01 = euc_dist(p0,p1)
            p2 = np.array((box_xcoord[2], box_ycoord[2]))
            p3 = np.array((box_xcoord[3], box_ycoord[3]))
            dist23 = euc_dist(p2,p3)
            dist = (dist01+dist23)/2
            box_length_all.append(dist)
    


        X_mid_all = []
        for b in center_coord:
            X_mid_all.append(b[0])

        Y_mid_all = []
        for b in center_coord:
            Y_mid_all.append(b[1])
            


        bounding_box_properties['angle_all'] = angle_all
        bounding_box_properties['center_coord'] = center_coord
        bounding_box_properties['boxcontent_center_all'] = boxcontent_center_all
        bounding_box_properties['box_length_all'] = box_length_all        
        bounding_box_properties['X_coord'] = X_mid_all
        bounding_box_properties['Y_coord'] = Y_mid_all    
        
            
        return bounding_box_properties  


    def extract_txt(self, bounding_boxes):
        
        txts = [line[1][0] for line in bounding_boxes]
        
        return txts





    






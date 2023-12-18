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
import yaml
from typing import List
import pandas as pd
import argparse
import PIL

paddle.utils.run_check()




class Document_structure():


    def __init__(self):
        pass


    def crop_with_jaro(self, fpath: str , path_yaml='/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/test_names.yaml') -> PIL.Image :

        """
        Given a jpeg of lab report, crop_with_jaro identifies the location of the table and crops that out and returns it as a jpeg.

        Args: 
        fpath: Path to original jpeg file
        path_yaml: yaml containing medical test names

        Returns: 
        Path to jpeg containing only the cropped image of lab report table.

        """

        with open(path_yaml, 'r') as file:
            tests = yaml.safe_load(file)
        test_list = tests['test_name']
        
        OCR_model = Paddle_OCR()
        Text_manip = Text_Manipulation()
        Bounding_box_comp = Boundingbox_Computations()
        Doc_preprocess = Doc_Preprocessor()

        bounding_boxes = OCR_model.apply_ocr(fpath)
        txts = Text_manip.extract_txt(bounding_boxes)
        bounding_box_properties = Bounding_box_comp.bounding_box_property_estimator(bounding_boxes)

        angle_correction = Bounding_box_comp.skew_calculate(bounding_box_properties)
        images_corrected = Doc_preprocess.rotate_img(fpath,angle_correction)

        Y_mid_all = bounding_box_properties['Y_coord']
        Jaro_Dist_All = Text_manip.jaro_comparison(txts,test_list)




        loc = list(np.where(np.array(Jaro_Dist_All) > 0.8))

        try:
            y_point_start = int(Y_mid_all[np.min(loc)])-20
            y_point_end = int(Y_mid_all[np.max(loc)])+20
        except:
            y_point_start = 0
            y_point_end = 800
        
        x_point_end = np.shape(images_corrected)[1]

        pt1_start = (0, y_point_start)
        pt2_start = (x_point_end, y_point_start)
        color = (0, 0, 0)
        abc = cv2.line(np.squeeze(np.array(images_corrected)), pt1_start, pt2_start,color, thickness = 3)

        pt1_start = (0, y_point_end)
        pt2_start = (x_point_end, y_point_end)
        color = (0, 0, 0)
        de = cv2.line(np.squeeze(np.array(abc)), pt1_start, pt2_start,color, thickness = 3)

        im_bottom = de[y_point_start:y_point_end,0:x_point_end]
        im_bottom = Image.fromarray(im_bottom)
        
        im_top = de[0:y_point_start,0:x_point_end]
        im_top = Image.fromarray(im_top)
        
        
        return im_bottom, im_top  



    def serialize_table(self, fpath:str, path_yaml='/home/nhadmin/users/sudarshan/doc_intelligence/data/lab_report/test_names.yaml') -> List:

        """
        Returns list of all row, column pairs in tables

        Args: 
        fpath: Path to original jpeg file
        path_yaml: yaml containing medical test names

        Returns: 
        list of all row, column pairs in tables


        """

        with open(path_yaml, 'r') as file:
            tests = yaml.safe_load(file)
        test_list = tests['test_name']


        OCR_model = Paddle_OCR()
        Text_manip = Text_Manipulation()
        Bounding_box_comp = Boundingbox_Computations()


        im_bottom, im_top = self.crop_with_jaro(fpath, path_yaml)

        bounding_boxes = OCR_model.apply_ocr('/home/nhadmin/users/bottom.jpg') 
        test_location = Text_manip.select_subset_boundingbox(bounding_boxes,test_list)
        pairwise_bounding_box_relations = Bounding_box_comp.inter_bounding_box_relations(test_location['jaro_matching'],bounding_boxes)
        serialized_table = Text_manip.row_extraction_overlap_method(pairwise_bounding_box_relations,test_location['jaro_matching'])
        print('Hello')

        return serialized_table


    def table_formation(self,key_pairs_all: list,path_yaml: str) -> pd.DataFrame:

        """
        Function returns dataframe containing medical test names, corresponding values and low and high range.

        Args: 
        key_pairs_all: List of candidate rows returned by row_extraction_slope_method
        path_yaml: path to yaml file containing medical test names.

        Returns:
        dataframe containing medical test names, corresponding values, low and high range


        """

        with open(path_yaml, 'r') as file:
            tests = yaml.safe_load(file)
        test_list = tests['test_name']

        d = pd.DataFrame()

        for ky in key_pairs_all:

          total_confidence = 0  
          count_confi = 0
          numb_entries = len(ky)
          entry = []
          key_value_table = ky[0][0]
          for entry_numb,kyz in enumerate(ky):      
              entry.append(ky[entry_numb][1:])

          test_name = ky[entry_numb][0]
          jaro_spec_all = []  
          jaro_dist_max = 0
          for t_spec in test_list:

                jaro_spec = jellyfish.jaro_distance(test_name.lower(),t_spec.lower())
                jaro_spec_all.append(jaro_spec)



                if jaro_spec > jaro_dist_max:
                   jaro_dist_max = jaro_spec
                   potential_test_name = t_spec


          if np.max(jaro_spec_all)<0.8:       
             test_name = 'NA'
          else:
             test_name = potential_test_name




          test_result = ["N/A"]
          test_flag = 0
          units = ["N/A"]
          units_flag = 0
          range = ["N/A"]    
          range_flag = 0
          max_dist = 100000
          min_overlap = -100000

          for i,kyz in enumerate(entry):


              min_angle = entry[-1][-1]
              if ('-' in entry[i][0]) or ('to' in entry[i][0]):


                  if range_flag == 1:
                    range_angle = entry[i][-1]
                    if abs(range_angle-min_angle)<abs(range_first_angle-min_angle):
                      range = entry[i][0]
                  else:
                    range = entry[i][0]
                    range_first_angle = entry[i][-1]
                    range_flag = 1

              elif ('/' in entry[i][0] or '%' in entry[i][0]) and not(any(chr.isdigit() for chr in str(entry[i][0]))):  

                  if units_flag == 1:   
                    units_angle = entry[i][-1]
                    if abs(units_angle-min_angle)<abs(units_first_angle-min_angle):
                      units = entry[i][0]
                  else:
                    units = entry[i][0]
                    units_first_angle = entry[i][-1]
                    units_flag = 1

              elif any(chr.isdigit() for chr in str(entry[i][0])):


                    test_dist = entry[i][-2]
                    if test_dist < max_dist:
                       max_dist = test_dist
                       test_result = str(entry[i][0])
                       confidence = entry[i][-4] 
                  






          Emp_str = []
          emp_str = ""

          Emp_str_corrected = []        
          emp_str_corrected = ""

          print('*******')
          print(test_name)
          print(test_result)
          print(units)
          print(range)

          for i,m in enumerate(range):
              if m.isdigit() or m=='.':

                        emp_str = emp_str + m
                        emp_str_corrected = emp_str_corrected + m



              elif not(m.isdigit()):

                        if emp_str:
                            Emp_str.append(emp_str) 
                        emp_str = ""

                        if emp_str_corrected:
                            Emp_str_corrected.append(emp_str_corrected) 
                        emp_str_corrected = ""                    

          if (test_result != ["N/A"]):

              if emp_str:
                Emp_str.append(emp_str)
              print(Emp_str)   
              if len(Emp_str)<2:
                temp = pd.DataFrame({'Test': [test_name],'Result': [test_result],'Confidence': [confidence],'Units': [units],'Low': 'NA','High': 'NA'})
                d = pd.concat([d, temp])
              elif len(Emp_str)>=2:
                try:
                  temp = pd.DataFrame({'Test': [test_name],'Result': [test_result],'Confidence': [confidence],'Units': [units],'Low': np.float(Emp_str[0]),'High': np.float(Emp_str[1])})
                  d = pd.concat([d, temp])
                except:
                   try:
                       temp = pd.DataFrame({'Test': [test_name],'Result': [test_result],'Confidence': [confidence],'Units': [units],'Low': 'NA','High': 'NA'})
                       d = pd.concat([d, temp])
                   except:
                       print('Nothing')
                       print(temp)         


        return d    




## crop with jaro

#obj = Document_structure()
#im_bottom, im_top  = obj.crop_with_jaro('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/labreport.jpeg', '/home/nhadmin/users/sudarshan/doc_intelligence/doc_intelligence/test_names.yaml')
#im_bottom.save('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/bottom/bottom.jpg')
#im_top.save('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/top/top.jpg')

## serialize table


obj = Document_structure()
serialized_table  = obj.serialize_table('/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/labreport.jpeg', '/home/nhadmin/users/sudarshan/doc_intelligence/data/table_crop/test_names.yaml')
print(serialized_table)

#obj = Document_structure()
#serialized_table  = obj.serialize_table('./data/table_crop/labreport.jpeg', './data/table_crop/test_names.yaml')
#print(serialized_table)

print('Hello')


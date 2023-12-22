
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
from utils.helper.computations import Boundingbox_Computations
from typing import List


paddle.utils.run_check()






class Text_Manipulation():

    def __init__(self):
        pass

    def extract_txt(self, bounding_boxes: list) -> list:

        """
        Returns list of texts in bounding boxes

        Args:
        bounding_boxes: List of bounding boxes

        Return:
        Returns list of texts in bounding boxes

        """

        txts = [line[1][0] for line in bounding_boxes]
        
        return txts

    def jaro_comparison(self, txts: list,test_list: list) -> List:

        """
        Returns list of texts in bounding boxes

        Args:
        txts: List of texts in bounding boxes
        test_list: list of medical test names

        Return:
        Returns list of Jaro distance values

        """

        
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


    
    def select_subset_boundingbox(self, bounding_boxes: list,test: list) -> List:

        """

        Given a list of bounding boxes returned by apply_ocr and list of common medical test names, this function will do string matching
        to find the relevant bounding boxes containing text closely matching to those provided in the test list 

        Args:

        bounding_boxes: List of bounding boxes 
        test: list of medical tests

        Returns:

        list of indices of the bounding boxes, whose text closely matches those in test list



        """
        
        textlocation = {}
        
        text_in_box_all = []
        jrd_all = []
        jrd_loc_all = []
        for counter,f in enumerate(bounding_boxes):

            text_in_box = f[1][0]
            jrd_max = -1

            for i,t in enumerate(test):
                jrd = jellyfish.jaro_distance(t.lower(),text_in_box.lower())
                if jrd > jrd_max:
                    jrd_max = jrd
                    loc = i 

            if jrd_max > 0.8:
                jrd_loc_all.append(counter)
                text_in_box_all.append(text_in_box)
                
        textlocation['text'] = text_in_box_all
        textlocation['jaro_matching'] = jrd_loc_all
                
        return textlocation  

    def row_extraction_overlap_method(self, pairwise_bounding_box_relations: list,jrd_loc_all:list) -> List:

        """
        Given a dictionary of interbounding box relations, row_extraction_overlap_method returns candidate rows based 
        on degree of overlap between bounding boxes

        Args: 
        pairwise_bounding_box_relations: dictionary containing relations between bounding boxes 
        jrd_loc_all:  list of Jaro distance values


        Returns: 
        list of rows

        """
    
        box_overlap_percentage = pairwise_bounding_box_relations['overlap_percentage']
        text_in_box = pairwise_bounding_box_relations['text_in_box']
    
        all_row = []
    
        for i,val in enumerate(box_overlap_percentage):
        
        
            key_loc = jrd_loc_all[i]
            key = text_in_box[0][key_loc]
            val_pass = list(np.array(text_in_box[0])[np.array(val)>0.5])
            print(val_pass)
            print(type(val_pass))
        
            for v_pass in val_pass:
            
                print('key and pass')
                print(key)
                print(v_pass)
                if key != v_pass:
                    new_row = [key, v_pass]
                    all_row.append(new_row)

            
        return  all_row



    def row_extraction_slope_method(self,pairwise_bounding_box_relations:list,test_list:list) -> List:

        """
        Given a dictionary of interbounding box relations, row_extraction_slope_method returns candidate rows
        based on slope between bounding boxes

        Args: 
        pairwise_bounding_box_relations: dictionary containing relations between bounding boxes 
        test_list: list of medical tests


        Returns: 
        list of rows

        """
        
        Bbox_comp = Boundingbox_Computations()

        key_pairs_all = []

        for i in range(len(pairwise_bounding_box_relations['key_text'])):   
            
            y_coord_diff_abs = abs(np.array(pairwise_bounding_box_relations['y_diff'][i]))
            y_coord_diff_abs_copy = y_coord_diff_abs.copy()
            y_coord_diff_abs_copy = y_coord_diff_abs_copy[~np.isnan(y_coord_diff_abs)]
            y_coord_diff_abs_copy = y_coord_diff_abs_copy[~np.isinf(y_coord_diff_abs_copy)]
            y_coord_diff_abs_copy_sorted = np.sort(y_coord_diff_abs_copy)
            y_coord_thresh = np.mean(y_coord_diff_abs_copy_sorted[1:10])
            
        #
            abs_slope = abs(np.array(pairwise_bounding_box_relations['y_diff'][i])/np.array(pairwise_bounding_box_relations['x_diff'][i]))
            abs_slope_copy = abs_slope.copy()
            abs_slope_copy = abs_slope_copy[~np.isnan(abs_slope)]
            abs_slope_copy = abs_slope_copy[~np.isinf(abs_slope_copy)]
            abs_slope_copy_sorted = np.sort(abs_slope_copy)
            abs_slope_thresh = np.mean(abs_slope_copy_sorted[0:10])
            

        #
            try:
                loc_miny_loc = list(np.squeeze(np.where(y_coord_diff_abs < y_coord_thresh)))
                loc_min_thresh = list(np.squeeze(np.where(abs_slope < abs_slope_thresh)))

                loc_miny_loc = Bbox_comp.intersection(loc_miny_loc,loc_min_thresh) 

                


            except:
                loc_miny_loc = []
                loc_min_thresh = []  


            min_dist = 100000000
            

            if np.size(loc_miny_loc):
            
                txt_temp =  pairwise_bounding_box_relations['key_text'][i][0]
                jaro_spec_all = []

                for t_spec in test_list:
                    
                        jaro_spec = jellyfish.jaro_distance(txt_temp.lower(),t_spec.lower())
                        jaro_spec_all.append(jaro_spec)

                if np.max(jaro_spec_all)<0.8:       
                    flag_proceed = 0
                else:
                    flag_proceed = 1

                if (flag_proceed):
                    key_pairs = []
                    key_val = []
                    for count, j in enumerate(loc_miny_loc):
                        
                        
                        if (pairwise_bounding_box_relations['text_in_box'][i][j] != pairwise_bounding_box_relations['key_text'][i][0]):


                            confidence =  pairwise_bounding_box_relations['overlap_percentage'][i][j]

                            box_dist = pairwise_bounding_box_relations['distance'][i][j] 
                            angle = pairwise_bounding_box_relations['angle'][i][j]

                            if box_dist <=min_dist:
                                min_angle = angle
                                min_dist = box_dist

                            key_val = [pairwise_bounding_box_relations['key_text'][i][0], pairwise_bounding_box_relations['text_in_box'][i][j], confidence, confidence, box_dist, angle, ]
                            key_pairs.append(key_val)
                    key_pairs.append([pairwise_bounding_box_relations['key_text'][i][0], 'MinAngle', min_angle])
                    key_pairs_all.append(key_pairs)
                    print(key_pairs_all)
                    
                    
            else:
                
                print('Empty')

        return key_pairs_all






        

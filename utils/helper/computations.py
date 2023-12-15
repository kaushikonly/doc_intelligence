
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




class Boundingbox_Computations():


    def __init__(self):
        pass


    def euc_dist(self, point1,point2):
        dist = np.linalg.norm(point1 - point2)
        return dist 

    def intersection(self,lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3


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
            dist01 = self.euc_dist(p0,p1)
            p2 = np.array((box_xcoord[2], box_ycoord[2]))
            p3 = np.array((box_xcoord[3], box_ycoord[3]))
            dist23 = self.euc_dist(p2,p3)
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


    def inter_bounding_box_relations(self, test_location,bounding_boxes):


        pairwise_bounding_box_relations = {}
        
        overlap_all = []
        overlap_vals_all = []
        slope_vals_all = []
        angle_vals_all = []
        distance_all = []
        y_diff_all = []
        x_diff_all = []
        key_text_all = []

        for loc in test_location:

            overlap_key = []
            overlap_vals = []
            slope_vals = []
            angle_vals = []
            distance = []
            y_diff = []
            x_diff = []

            y_min = np.min([bounding_boxes[loc][0][0][1],bounding_boxes[loc][0][1][1],bounding_boxes[loc][0][2][1],bounding_boxes[loc][0][3][1]])
            y_max = np.max([bounding_boxes[loc][0][0][1],bounding_boxes[loc][0][1][1],bounding_boxes[loc][0][2][1],bounding_boxes[loc][0][3][1]])

            x_min = np.min([bounding_boxes[loc][0][0][0],bounding_boxes[loc][0][1][0],bounding_boxes[loc][0][2][0],bounding_boxes[loc][0][3][0]])
            x_max = np.max([bounding_boxes[loc][0][0][0],bounding_boxes[loc][0][1][0],bounding_boxes[loc][0][2][0],bounding_boxes[loc][0][3][0]])

            y_avg = (y_min + y_max)/2
            x_avg = (x_min + x_max)/2


            for counter,f in enumerate(bounding_boxes):

                #if counter != loc:

                y_min_comp = np.min([bounding_boxes[counter][0][0][1],bounding_boxes[counter][0][1][1],bounding_boxes[counter][0][2][1],bounding_boxes[counter][0][3][1]])
                y_max_comp = np.max([bounding_boxes[counter][0][0][1],bounding_boxes[counter][0][1][1],bounding_boxes[counter][0][2][1],bounding_boxes[counter][0][3][1]])

                x_min_comp = np.min([bounding_boxes[counter][0][0][0],bounding_boxes[counter][0][1][0],bounding_boxes[counter][0][2][0],bounding_boxes[counter][0][3][0]])
                x_max_comp = np.max([bounding_boxes[counter][0][0][0],bounding_boxes[counter][0][1][0],bounding_boxes[counter][0][2][0],bounding_boxes[counter][0][3][0]])


                y_comp_avg = (y_min_comp + y_max_comp)/2
                x_comp_avg = (x_min_comp + x_max_comp)/2

                if (y_min <= y_min_comp) and (y_max >= y_max_comp): 
                    overlap = 1
                elif (y_min >= y_min_comp) and (y_max <= y_max_comp): 
                    overlap = 1            
                elif (y_min >= y_min_comp) and (y_max >= y_max_comp): 
                    overlap =  (y_max_comp -y_min)/(y_max -y_min)
                elif (y_min <= y_min_comp) and (y_max <= y_max_comp): 
                    overlap =  (y_max -y_min_comp)/(y_max -y_min) 
                else:
                    print('No option')

                slope =   np.abs((y_comp_avg - y_avg)/(x_comp_avg - x_avg))
                angle = math.degrees(math.atan(slope))

                dist = math.sqrt(((y_comp_avg-y_avg)**2)+((x_comp_avg-x_avg)**2))

                ydiff = y_avg-y_comp_avg
                xdiff = x_avg-x_comp_avg

                if overlap <=0:
                        #overlap = 0 
                        overlap_val = 'NaN'       
                        overlap_val = bounding_boxes[counter][1][0]


                else:
                        overlap_val = bounding_boxes[counter][1][0]

                overlap_key.append(overlap)        
                overlap_vals.append(overlap_val)
                slope_vals.append(slope)
                angle_vals.append(angle)
                distance.append(dist)
                y_diff.append(ydiff)
                x_diff.append(xdiff)


            overlap_all.append(overlap_key)            
            overlap_vals_all.append(overlap_vals)      
            slope_vals_all.append(slope_vals)
            angle_vals_all.append(angle_vals)
            distance_all.append(distance)
            y_diff_all.append(y_diff)
            x_diff_all.append(x_diff)
            key_text_all.append(bounding_boxes[loc][1])
            
            pairwise_bounding_box_relations['overlap_percentage'] = overlap_all
            pairwise_bounding_box_relations['text_in_box'] = overlap_vals_all
            pairwise_bounding_box_relations['slope'] = slope_vals_all
            pairwise_bounding_box_relations['angle'] = angle_vals_all
            pairwise_bounding_box_relations['distance'] = distance_all
            pairwise_bounding_box_relations['y_diff'] = y_diff_all
            pairwise_bounding_box_relations['x_diff'] = x_diff_all
            pairwise_bounding_box_relations['key_text'] = key_text_all

        return pairwise_bounding_box_relations






    






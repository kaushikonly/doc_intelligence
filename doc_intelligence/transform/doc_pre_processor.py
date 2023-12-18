import os
import cv2
import json
import math
import PIL
import copy
import jellyfish
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from doc_intelligence.model.ocr.paddleocr import Paddle_OCR

class Doc_Preprocessor():

    """
    Containts the following document processing techniqes, 
    - Deskew
    - Shadow Remover
    - Line Remover
    - Dewarp 
    
    """

    def __init__(self) -> None:

        self.paddleocr_model = Paddle_OCR()

    def __skew_calculate(self, coor_list : list):

        """
        Compute the slope of the bounding boxes and return the median value 

        Args: 
            coor_list (list): List of list where each element contains 4 coordinates of the bounding box
        Returns: 
            Mean or median value of the slop of the all the bounding boxes
        
        """

        angle_all = []

        for ind in range(len(coor_list)):
            box_coord = np.array(coor_list[ind])
            box_xcoord = box_coord[0:4, 0]
            box_ycoord = box_coord[0:4, 1]

            slope_1 = (box_ycoord[1] - box_ycoord[0]) / (box_xcoord[1] - box_xcoord[0])
            slope_2 = (box_ycoord[2] - box_ycoord[3]) / (box_xcoord[2] - box_xcoord[3])

            slope_mean = (slope_1 + slope_2) / 2
            angle = math.degrees(math.atan(slope_mean))
            angle_all.append(angle)

        if np.median(angle_all) == 0:
            angle_correction = np.mean(angle_all)
        else:
            angle_correction = np.median(angle_all)

        return abs(angle_correction)

    def deskew_image(self, image_path: str) -> PIL.Image: 

        """
        Deskew the image using the slop of the bouding boxes 

        Args: 
            image_path (str): path of the image
        Returns: 
            Deskewed PIL image
        
        """

        image = Image.open(image_path).convert('RGB')
        flat_results = self.paddleocr_model.apply_ocr(image_path)

        self.angle_correction =  self.__skew_calculate([value[0] for value in flat_results])

        images_corrected = image.rotate(
            self.angle_correction, resample=Image.BICUBIC, expand=True
        )

        return images_corrected

    def shadow_remove(self, image_path: str) -> PIL.Image:
        """ Remove the shadow from the document image
        
        Args: 
            image_path: path to the document image
        Returns: 
            Pil image after removing the shadow from the document image
        """
        rgb_planes = cv2.split(cv2.imread(image_path))

        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(
                diff_img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            )
            result_norm_planes.append(norm_img)
        img_shadow_removed = cv2.merge(result_norm_planes)

        color_converted = cv2.cvtColor(img_shadow_removed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(color_converted)

    def remove_horz_verti_lines(self, image_path: str, vh_thresh : int = 200, hoiz_line: bool = True, verti_line: bool= False) -> PIL.Image:

        """
        To remove eighter horizontal or vertical (or both) lines from the document image

        Args: 
            image_path: path to the image
            vh_thresh: threshold for the binarization, range [0, 255], default = 200
            hoiz_line: bool flag for removing horizontal lines
            verti_line: boolean flag for removing vertical lines 

        Returns: 
            PIL image after removing lines
        """
        # read image using cv2 
        image = cv2.imread(image_path)

        if len(image.shape) != 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = copy.deepcopy(image)

        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        gray = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        if hoiz_line:
            horizontal = np.copy(bw)

            cols = horizontal.shape[1]
            horizontal_size = cols // 30

            # Create structure element for extracting horizontal lines through morphology operations
            horizontalStructure = cv2.getStructuringElement(
                cv2.MORPH_RECT, (horizontal_size, 1)
            )
            # Apply morphology operations
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

        if verti_line: 
            vertical = np.copy(bw)

            # Specify size on vertical axis
            rows = vertical.shape[0]
            verticalsize = rows // 30

            # Create structure element for extracting vertical lines through morphology operations
            verticalStructure = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, verticalsize)
            )
            # Apply morphology operations
            vertical = cv2.erode(vertical, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)

        shadowless_vh = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        ) 

        shadowless_vh[shadowless_vh > vh_thresh] = 255
        shadowless_vh[shadowless_vh <= vh_thresh] = 0
        shadowless_vh_inv = np.invert(shadowless_vh)

        shadowless_vhless_inv = shadowless_vh_inv - horizontal if hoiz_line else shadowless_vh_inv
        shadowless_vhless_inv = shadowless_vh_inv - vertical if verti_line else shadowless_vh_inv
        shadowless_vhless_pil = Image.fromarray(np.invert(shadowless_vhless_inv))

        return shadowless_vhless_pil

    def image_crop(self, shadowless_vhless_inv):
        shadowless_vhless_inv_sum_y = np.sum(shadowless_vhless_inv, axis=0)
        plt.plot(shadowless_vhless_inv_sum_y)

        thresh = np.median(shadowless_vhless_inv_sum_y) / 4
        thresh_all = len(shadowless_vhless_inv_sum_y) * [thresh]

        plt.plot(thresh_all)
        plt.show()
        loc_thresh = np.squeeze(
            np.where(shadowless_vhless_inv_sum_y >= thresh_all)
        )
        print(loc_thresh)
        thresh_diff = np.diff(loc_thresh)
        print(thresh_diff)
        thresh_diff_grt1 = np.squeeze(np.where(thresh_diff > 1))
        print(thresh_diff_grt1)
        histogram_area = np.diff(thresh_diff_grt1)
        print(histogram_area)

        loc_max = np.squeeze(
            np.where(histogram_area == np.max(histogram_area))
        )

        crop_start = loc_thresh[thresh_diff_grt1[loc_max] - 1]
        crop_end = loc_thresh[thresh_diff_grt1[loc_max + 2]]

        shadowless_vhless_inv_sum_y = np.sum(shadowless_vhless_inv, axis=0)
        plt.plot(shadowless_vhless_inv_sum_y[crop_start:crop_end])
        plt.show()

        shadowless_vhless_inv_crop = shadowless_vhless_inv[
            :, crop_start:crop_end
        ]

        shadowless_vhless_inv_crop_pil = Image.fromarray(
            shadowless_vhless_inv_crop
        )
        shadowless_vhless_inv_crop_pil.show()

        shadowless_vhless_inv_crop_inv = np.invert(shadowless_vhless_inv_crop)
        shadowless_vhless_inv_crop_inv_pil = Image.fromarray(
            shadowless_vhless_inv_crop_inv
        )
        shadowless_vhless_inv_crop_inv_pil.show()

        return shadowless_vhless_inv_crop, shadowless_vhless_inv_crop_pil

    def crop_with_jaro(self, test_list, txts, Y_mid_all, img, counter):
        Jaro_Dist_All = []

        for indiv_txt in txts:
            Jaro_Dist = []

            for indiv_test in test_list:
                indiv_txt = indiv_txt.lower()
                indiv_test = indiv_test.lower()
                jaro_dist = jellyfish.jaro_distance(indiv_txt, indiv_test)
                Jaro_Dist.append(jaro_dist)
            Jaro_Dist_All.append(np.max(Jaro_Dist))

        loc = list(np.where(np.array(Jaro_Dist_All) > 0.8))
        print(loc)
        print(np.shape(loc))
        print(type(loc))
        try:
            y_point_start = int(Y_mid_all[np.min(loc)]) - 20
            y_point_end = int(Y_mid_all[np.max(loc)]) + 20
        except:
            y_point_start = 0
            y_point_end = 800

        x_point_end = np.shape(img)[1]

        pt1_start = (0, y_point_start)
        pt2_start = (x_point_end, y_point_start)
        color = (0, 0, 0)
        abc = cv2.line(
            np.squeeze(np.array(img)), pt1_start, pt2_start, color, thickness=3
        )

        pt1_start = (0, y_point_end)
        pt2_start = (x_point_end, y_point_end)
        color = (0, 0, 0)
        de = cv2.line(
            np.squeeze(np.array(abc)), pt1_start, pt2_start, color, thickness=3
        )
        plt.imshow(de)

        im = de[y_point_start:y_point_end, 0:x_point_end]
        im_top = de[0:y_point_start, 0:x_point_end]
        plt.imshow(im)

        filepath = "/data/app/table_ocr/"

        filename = "Cropped_image_" + str(counter) + ".jpg"

        full_filepath = os.path.join(filepath, filename)

        im = Image.fromarray(im)
        im.save(full_filepath)

        Load_img = load_images_functions()
        img = Load_img.load_images_from_folder(full_filepath)

        im_top = de[0:y_point_start, 0:x_point_end]

        filepath = "/data/app/table_ocr/"
        filename = "Cropped_image_top_" + str(counter) + ".jpg"

        full_filepath_top = os.path.join(filepath, filename)

        im_top = Image.fromarray(im_top)
        im_top.save(full_filepath_top)
        im_top.show()

        return img, full_filepath, im_top, full_filepath_top
    
    def segmentation(self, center_coord):
        X_diff = []
        Y_diff = []
        X_coord = []
        Y_coord = []
        X_mid_all = []
        Y_mid_all = []

        for i, box_coordi in enumerate(center_coord):
            X_mid_all.append(np.squeeze(box_coordi)[0])
            Y_mid_all.append(np.squeeze(box_coordi)[1])

            if i == 0:
                store_prev = np.squeeze(box_coordi)
            else:
                x_diff = np.squeeze(box_coordi)[0] - store_prev[0]
                y_diff = np.squeeze(box_coordi)[1] - store_prev[1]

                X_diff.append(x_diff)
                Y_diff.append(y_diff)
                X_coord.append(np.squeeze(box_coordi)[0])
                Y_coord.append(np.squeeze(box_coordi)[1])

                store_prev = np.squeeze(box_coordi)

        return X_coord, Y_coord, X_mid_all, Y_mid_all

    def rotate_img(self,img_path,angle_correction):

        """
        Returns deskewd image

        Args:
        Path to skewed image and skew of document, which is estimated by skew_calculate

        Returns:
        Deskewed image
        """   
        img = Image.open(img_path)
        images_corrected = img.rotate(angle_correction,resample=Image.BICUBIC, expand=True)
        images_corrected = images_corrected.convert('RGB')
          
        return images_corrected

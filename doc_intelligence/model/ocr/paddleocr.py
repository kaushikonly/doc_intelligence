
import os
import cv2
import copy
import math
import array
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFont
from typing import List

# from ml.src.utils.helper.image_process_calculations import Process_Image


class Paddle_OCR(PaddleOCR):

    def __init__(self, language="en"):
        self.language = language
        self.ocr_model = PaddleOCR(lang=self.language)

    def apply_ocr(self, img_path: str) -> List:

        """
        locates text in image, and makes the text machine readable

        Args:
        img_path: Path to .jpeg file

        Returns:
        list of OCR'ed text with bounding box coordinates and confidence score
        """

        result = self.ocr_model.ocr(img_path, cls=True)
        flat_result = [item for sublist in result for item in sublist]

        self.boxes = [line[0] for line in flat_result]
        self.texts = [line[1][0] for line in flat_result]
        self.scores = [line[1][1] for line in flat_result]

        return flat_result

    def ocr_img(self, img_path):

        ocr = PaddleOCR(
            use_angle_cls=True, lang="en"
        )  # need to run only once to download and load model into memory
        result = ocr.ocr(img_path, cls=True)
        flat_result = [item for sublist in result for item in sublist]
        image = Image.open(img_path).convert("RGB")
        boxes = [line[0] for line in flat_result]
        txts = [line[1][0] for line in flat_result]
        scores = [line[1][1] for line in flat_result]
        # font = ImageFont.load_default()
        image_array = Image.open(img_path)

        return image, flat_result, image_array, txts

    def draw_multiple_boxes(self, image, ind_list):

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
        ocr = PaddleOCR(
            use_angle_cls=True, lang="en"
        )  # need to run only once to download and load model into memory
        result = ocr.ocr(img_path, det=True, rec=False, cls=True)

        flat_result = [item for sublist in result for item in sublist]
        image = Image.open(img_path).convert("RGB")
        boxes = [line[0] for line in flat_result]
        image_array = Image.open(img_path)

        return image, flat_result, image_array

    def coord_angle(self, img, flat_result):
        all_coord = []
        for i in range(len(flat_result)):
            coord = flat_result[i]
            all_coord.append(coord)

        all_coord_rev = all_coord[::-1]

        angle_all = []

        Coord_new_all = []

        proc_img = Process_Image()

        for i, coord in enumerate(all_coord_rev):
            slope1, c_avg1 = proc_img.slope_c(
                coord[0][0], coord[1][0], coord[0][1], coord[1][1]
            )
            slope2, c_avg2 = proc_img.slope_c(
                coord[2][0], coord[3][0], coord[2][1], coord[3][1]
            )

            y1 = c_avg1
            y4 = c_avg2
            x1 = 0
            x4 = 0

            y2 = slope1 * img.shape[1] + c_avg1
            y3 = slope2 * img.shape[1] + c_avg2
            x2 = img.shape[1]
            x3 = img.shape[1]

            coord_new = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            Coord_new_all.append(coord_new)

            angle_all.append(
                np.mean(
                    [
                        math.degrees(math.atan(slope1)),
                        math.degrees(math.atan(slope2)),
                    ]
                )
            )

        return angle_all, all_coord_rev, all_coord, Coord_new_all

    def intersect_boxes_percentage(self, Coord_new_all):
        area_intersect_percentage_all = []

        for i, coord in enumerate(Coord_new_all):
            area_intersect_percentage = []
            for j, coord_compare in enumerate(Coord_new_all):
                edge_point = [
                    (coord[0][0], coord[0][1]),
                    (coord[1][0], coord[1][1]),
                    (coord[2][0], coord[2][1]),
                    (coord[3][0], coord[3][1]),
                ]
                edge_point_compare = [
                    (coord_compare[0][0], coord_compare[0][1]),
                    (coord_compare[1][0], coord_compare[1][1]),
                    (coord_compare[2][0], coord_compare[2][1]),
                    (coord_compare[3][0], coord_compare[3][1]),
                ]

                p = Polygon(edge_point)
                q = Polygon(edge_point_compare)

                if p.intersects(q):
                    area_intersect = p.intersection(q).area
                    area_base = p.area
                    area_compare = q.area

                    intersect_percentage = int(
                        100 * (area_intersect / area_base)
                    )
                    area_intersect_percentage.append(intersect_percentage)

                elif not (p.intersects(q)):
                    area_intersect_percentage.append(0)

            area_intersect_percentage_all.append(area_intersect_percentage)

        return area_intersect_percentage_all

    def skew_calculate(self, flat_result):
        center_coord = []
        boxcontent_center_all = []
        boxcontent_coord_all_x = []
        boxcontent_coord_all_y = []
        text_all = []
        slope_all = []
        angle_all = []

        for i, bla in enumerate(flat_result):
            box_coord = np.array(flat_result[i][0])
            text_all.append(flat_result[i][1][0])
            box_xcoord = box_coord[0:4, 0]
            box_ycoord = box_coord[0:4, 1]

            boxcenter_xcoord = np.mean(box_coord[0:4, 0])
            boxcenter_ycoord = np.mean(box_coord[0:4, 1])
            slope_1 = (box_ycoord[1] - box_ycoord[0]) / (
                box_xcoord[1] - box_xcoord[0]
            )
            slope_2 = (box_ycoord[2] - box_ycoord[3]) / (
                box_xcoord[2] - box_xcoord[3]
            )

            slope_mean = (slope_1 + slope_2) / 2
            angle = math.degrees(math.atan(slope_mean))
            angle_all.append(angle)
            box_center = array.array("d", [boxcenter_xcoord, boxcenter_ycoord])
            boxcontent_center = [flat_result[i][1][0], box_center]
            boxcontent_center_all.append(boxcontent_center)
            center_coord.append(box_center)
            boxcontent_coord_all_x.append(box_coord[0:4, 0])
            boxcontent_coord_all_y.append(box_coord[0:4, 1])
            slope_all.append(slope_mean)

        return angle_all, center_coord, boxcontent_center_all



    def ocr_img_sud(self, img_path):
    
        result = self.ocr(img_path, cls=True)
        flat_result = [item for sublist in result for item in sublist]
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in flat_result]
        txts = [line[1][0] for line in flat_result]
        scores = [line[1][1] for line in flat_result]
        font = ImageFont.load_default()
        image_array = Image.open(img_path)

        #nothing

        return flat_result
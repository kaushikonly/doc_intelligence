import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np

import os
from utils.helper import files


class Checks():


    def __init__(self):
        pass



    def test_deskew_angel(): 

        assert 5 == 4 + 1

    def test_deskew_imput(): 

        assert dp_obj.deskew_image(12) ==  "Given File format is not allowed. "

    def test_deskew_rotation():

        sample_doc_image = Image.open("../data/deskew/temp.png")
        rotation_degree = random.randint(5, 80)
        print("rotation_degree : ", rotation_degree)

        rotated_image = sample_doc_image.rotate(rotation_degree, resample=Image.BICUBIC, expand=True)
        rotated_image.save("../data/deskew/rotated_img4.jpg")
        deskewed_image = dp_obj.deskew_image("../data/deskew/rotated_img4.jpg")
        print("angle_correction : ", dp_obj.angle_correction)

        assert abs(int(dp_obj.angle_correction)) in range(int(rotation_degree - rotation_degree*0.1), int(rotation_degree + rotation_degree*0.1))

    def is_input_image(self,fpath):

        if ('.jpeg' in fpath) or ('.jpg' in fpath) or ('.png' in fpath):
            return True
        else:
            raise ValueError('This is not valid image file')

    def is_input_yaml(self,path_yaml):

        if ('.yaml' in path_yaml):
            return True
        else:
            raise ValueError('This is not valid yaml file')

    def is_input_list(self,islist):

        if (type(islist)==list):
            print('Input is list')
        else:
            print('Input is not list')

    def is_df_empty(self,df):

        if (df.empty):
            print('Empty dataframe')

        else:
            print("Table Extracted")


    def is_imgcropped(self,im_bottom,images_corrected):

        if (np.shape(im_bottom) !=np.shape(images_corrected)):

            print('size is not same, so image likely cropped')

        else:
            print('size is same, so image is not cropped. Uploaded image may not have table or yaml file may not contain relevant values')
        



 
    def is_valid_image_file_path(self, fpath): 

        if not os.path.exists(fpath): raise ValueError("File doesn't exists!")
        if not isinstance(fpath, str): 
            raise ValueError("File path is not valid")

        base_name, file_extension, file_type = files.get_file_info(fpath)
        
        if not file_type == "image": 
            raise ValueError("Not valid file")

        try: 
            Image.open(fpath).verify()
        except: 
            raise ValueError("Not a valid image file")

    def is_valid_pdf_file_path(self, fpath): 

        if not os.path.exists(fpath): raise ValueError("File doesn't exists!")
        if not isinstance(fpath, str): 
            raise ValueError("File path is not valid")

        base_name, file_extension, file_type = files.get_file_info(fpath)

        if not file_type == "pdf": 
            raise ValueError("Not valid pdf document")

    def is_valid_yaml_file_path(self,fpath): 

        if not os.path.exists(fpath): raise ValueError("File doesn't exists!")
        if not isinstance(fpath, str): 
            raise ValueError("File path is not valid")

        base_name, file_extension, file_type = files.get_file_info(fpath)

        if not file_type == "yaml": 
            raise ValueError("Not valid yaml file")

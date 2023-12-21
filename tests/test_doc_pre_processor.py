import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np
from extract.structure import Document_structure

# get instance
dp_obj = dp.Doc_Preprocessor()

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

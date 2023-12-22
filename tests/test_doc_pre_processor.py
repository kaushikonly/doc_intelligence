import PIL
import pytest
import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp

# get instance
dp_obj = dp.Doc_Preprocessor()

def test_deskew_invliad_image_path(): 

    with pytest.raises(ValueError):
        dp_obj.deskew_image(15)

def test_deskew_invliad_pdf_path(): 

    with pytest.raises(ValueError):
        dp_obj.deskew_image("./data/document_question_answering/discharge_summary_pdf_example.pdf")

def test_wrong_image_path(): 

    with pytest.raises(ValueError):
        dp_obj.deskew_image("./data/document_question_answering/wrong_file.jpg")

def test_deskew_rotation():

    sample_doc_image = Image.open("./data/deskew/temp.png")
    rotation_degree = random.randint(5, 80)
    rotation_degree = 30
    print("rotation_degree : ", rotation_degree)

    rotated_image = sample_doc_image.rotate(rotation_degree, resample=Image.BICUBIC, expand=True)
    rotated_image.save("./data/deskew/rotated_img4.jpg")
    deskewed_image = dp_obj.deskew_image("./data/deskew/rotated_img4.jpg")
    print("angle_correction : ", dp_obj.angle_correction)

    assert abs(int(dp_obj.angle_correction)) in range(int(rotation_degree - rotation_degree*0.2), int(rotation_degree + rotation_degree*0.2))

def test_shadow_remove_pdf_path(): 

    with pytest.raises(ValueError):
        dp_obj.shadow_remove("./data/document_question_answering/discharge_summary_pdf_example.pdf")

def test_shadow_remover_end_to_end(): 

    processed_img = dp_obj.line_remover("./data/shadow_remover/test_image_1.png", hori_line=True)
    processed_img.verify() 

def test_line_remover_pdf_path(): 

    with pytest.raises(ValueError):
        dp_obj.line_remover("./data/document_question_answering/discharge_summary_pdf_example.pdf")


def test_line_remover_end_to_end(): 

    processed_img = dp_obj.line_remover("./data/line_remover/test_img_2.png", hori_line=True)
    processed_img.verify() 

def test_line_remover_both_flase(): 

    with pytest.raises(ValueError):
        dp_obj.line_remover("./data/line_remover/test_img_2.png", hori_line=False, verti_line=False)



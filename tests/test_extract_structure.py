import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np
from extract.structure import Document_structure
import pytest

# get instance
doc_struct = Document_structure()


def test_crop_with_jaro_yaml():

    with pytest.raises(ValueError):
        doc_struct.crop_with_jaro('file1.img','file2.pdf')

def test_serialize_table_yaml():

    with pytest.raises(ValueError):
        doc_struct.serialize_table('file1.img','file2.pdf')

def test_imput_image_file():

    with pytest.raises(ValueError):
        doc_struct.crop_with_jaro('file1.pdf',"/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/test_names.yaml")

def test_crop_with_jaro_end_to_end(): 

    im_bottom, im_top = doc_struct.crop_with_jaro("/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/LabReport.jpeg", "/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/test_names.yaml")

    im_bottom.verify()
    im_top.verifty()

def test_serialize_table_pdf(): 

    with pytest.raises(ValueError):
        doc_struct.serialize_table('file1.pdf',"/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/test_names.yaml")

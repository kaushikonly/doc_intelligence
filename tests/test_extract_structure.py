import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np
from extract.structure import Document_structure
import pytest
import sys,os 
sys.path.append(os.path.realpath('..'))
# get instance
doc_struct = Document_structure()


def test_input_yaml():

    
    with pytest.raises(ValueError):
        result = doc_struct.crop_with_jaro('file1.img','file2.pdf')

def test_imput_image_file():

    with pytest.raises(ValueError):
        result = doc_struct.crop_with_jaro('file1.pdf','./data/lab_report/test_names.yaml')

def test_extract_table_yaml():

    with pytest.raises(ValueError):
        doc_struct.serialize_table('./data/lab_report/LabReport.jpeg','file2.pdf')

def test_extract_table_image_file():

    with pytest.raises(ValueError):
        doc_struct.crop_with_jaro('file1.pdf',"./data/lab_report/test_names.yaml")

def test_extract_table_end_to_end(): 

     table_df = doc_struct.crop_with_jaro("./data/lab_report/LabReport.jpeg", "./data/lab_report/test_names.yaml")


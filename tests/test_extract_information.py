import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np
from extract.information import Document_Information
import pytest
import sys,os 
sys.path.append(os.path.realpath('..'))
# get instance
doc_info= Document_Information()

def test_extract_table_yaml():

    with pytest.raises(ValueError):
        doc_info.extract_table('./data/lab_report/LabReport.jpeg','file2.pdf')

def test_extract_table_image_file():

    with pytest.raises(ValueError):
        doc_info.extract_table('file1.pdf',"./data/LabReport/test_names.yaml")

def test_extract_table_end_to_end(): 

    table_df = doc_info.extract_table("./data/lab_report/LabReport.jpeg", "./data/lab_report/test_names.yaml")
    assert table_df.shape[0] != 0

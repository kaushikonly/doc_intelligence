import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np
from extract.structure import Document_Information
import pytest

# get instance
doc_info= Document_Information()

def test_extract_table_yaml():

    with pytest.raises(ValueError):
        doc_info.serialize_table('file1.img','file2.pdf')

def test_extract_table_image_file():

    with pytest.raises(ValueError):
        doc_info.crop_with_jaro('file1.pdf',"/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/test_names.yaml")

def test_extract_table_end_to_end(): 

    table_df = doc_info.crop_with_jaro("/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/LabReport.jpeg", "/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/test_names.yaml")

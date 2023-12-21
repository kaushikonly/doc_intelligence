import random
from PIL import Image
from doc_intelligence.transform import doc_pre_processor as dp
import numpy as np
from extract.structure import Document_structure
import pytest
# get instance
doc_struct = Document_structure()


def test_input_yaml():

    
    with pytest.raises(ValueError):
        result = doc_struct.crop_with_jaro('file1.img','file2.pdf')

def test_imput_image_file():

    with pytest.raises(ValueError):
        result = doc_struct.crop_with_jaro('file1.pdf','/Users/sudarshansekhar/work/doc_intelligence/data/lab_report/test_names.yaml')
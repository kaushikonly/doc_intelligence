import os
from PIL import Image
from utils.helper import files

 
def is_valid_image_file_path(fpath): 

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

def is_valid_pdf_file_path(fpath): 

    if not os.path.exists(fpath): raise ValueError("File doesn't exists!")
    if not isinstance(fpath, str): 
        raise ValueError("File path is not valid")

    base_name, file_extension, file_type = files.get_file_info(fpath)

    if not file_type == "pdf": 
        raise ValueError("Not valid pdf document")

def is_valid_yaml_file_path(fpath): 

    if not os.path.exists(fpath): raise ValueError("File doesn't exists!")
    if not isinstance(fpath, str): 
        raise ValueError("File path is not valid")

    base_name, file_extension, file_type = files.get_file_info(fpath)

    if not file_type == "yaml": 
        raise ValueError("Not valid yaml file")


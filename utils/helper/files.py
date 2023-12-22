import json
import os
import tempfile
import uuid
from io import BytesIO
from typing import Optional, Tuple

import yaml
from fastapi import status
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


def save_file(file) -> str:
    """Save file object to the temp directory and return path"""
    _, file_extension, file_type = get_file_info(file.filename)

    if file_extension is None:
        raise ValueError("File not found")

    # Save file to temp directory
    temp_file_path = get_temp_file_path(file.filename)
    print("Saving file to {}".format(temp_file_path))

    if file_type == "image":
        # Convert into PIL image
        image = Image.open(BytesIO(file.file.read()))
        image.save(temp_file_path)
    else:
        with open(temp_file_path, "wb") as file_object:
            file_object.write(file.read())

    return temp_file_path


def get_temp_file_path(filename: str) -> str:
    """Generate a unique temp file path"""
    base_name, extension, _ = get_file_info(filename)
    random_suffix = str(uuid.uuid4())[:8]
    unique_filename = secure_filename(
        f"{base_name}_{random_suffix}{extension}"
    )
    return os.path.join(tempfile.gettempdir(), unique_filename)


def get_file_info(filename: str) -> Tuple[str, str, Optional[str]]:
    """Return base name, extension, and file type"""
    base_name, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()
    file_type = (
        "pdf"
        if file_extension == ".pdf"
        else "image"
        if file_extension in ALLOWED_EXTENSIONS
        else "yaml"
        if file_extension == ".yaml" or file_extension == ".yml"
        else None
    )
    return base_name, file_extension, file_type


def read_yaml_file(file_path):
    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def convert_yaml_to_json(config):
    config_json = json.dumps(config, indent=2)
    config_json = json.loads(config_json)
    return config_json

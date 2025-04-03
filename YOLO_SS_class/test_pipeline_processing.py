import os
import pytest
import shutil
import geopandas as gpd
import rasterio
import numpy as np
from pathlib import Path
from pipeline_processing import YOLOPreprocessor  # Update with actual filename



TEST_BASE_PATH = "test_data/"
TEST_GRID = "yolo_semi_janv_2025/grid_320_semi.shp"
TEST_TIF = "Pleiades_Vue1_2023/C1_orthoimage_forward.tif"
TEST_SHAPE = "TreeSample_ImagePleiade14feb2023_Pansharpen.shp"


@pytest.fixture(scope="module")
def pipeline():
    """Fixture to initialize and clean up test pipeline."""
    if os.path.exists(TEST_BASE_PATH):
        shutil.rmtree(TEST_BASE_PATH)  # Clean before testing
    pipeline = YOLOPreprocessor(TEST_BASE_PATH)
    yield pipeline
    shutil.rmtree(TEST_BASE_PATH)  # Clean up after testing

def test_directory_creation(pipeline):
    """Test that necessary directories are created."""
    for path in pipeline.paths.values():
        assert os.path.exists(path), f"Directory {path} was not created"

def test_extract_images(pipeline):
    """Test the extraction of images from a TIFF."""
    pipeline.extract_images(TEST_GRID, TEST_TIF)
    extracted_images = list(Path(pipeline.paths["image_set"]).glob("*.tif"))
    assert len(extracted_images) > 0, "No images were extracted"

def test_separate_grid(pipeline):
    """Test separation of labeled and unlabeled images."""
    pipeline.separate_grid(TEST_SHAPE)
    labeled_images = list(Path(pipeline.paths["image_label"]).glob("*.tif"))
    unlabeled_images = list(Path(pipeline.paths["image_unlabel"]).glob("*.tif"))
    assert len(labeled_images) + len(unlabeled_images) > 0, "No images were separated"
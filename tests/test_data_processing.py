import pytest
import pandas as pd
from cleave_app.data_processing import DataCollector
from io import BytesIO
from PIL import Image
import numpy as np

@pytest.fixture
def mock_image():
    fake_image = BytesIO()
    image = Image.new("RGB", (100, 100))
    image.save(fake_image, format="PNG")
    fake_image.seek(0)
    return fake_image

def test_data_collector_full(tmp_path, mock_image):
    img_folder = tmp_path / "images"
    img_folder.mkdir()
    image_path = img_folder / "image1.png"
    with open(image_path, "wb") as f:
        f.write(mock_image.read())

    csv_content = f"""ImagePath,FiberType,DateCreated,Diameter,CleaveAngle,CleaveTension,TensionVelocity,FHBOffset,ScribeDiameter,Misting,Hackle,Tearing
image1.png,PM15U25d,2025-06-09 15:37,123.5,0.22,193,60,2552,17.28,0,0,1
"""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(csv_content)

    collector = DataCollector(csv_path=str(csv_path), img_folder=str(img_folder) + "/")

    # 4. Check output
    assert collector.df is not None
    assert collector.df['CleaveQuality'].tolist() == [1]
    assert isinstance(collector.df['ImagePath'][0], str)
    assert collector.df['ImagePath'][0] == "image1.png"

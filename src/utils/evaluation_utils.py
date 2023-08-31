from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import os
import re

from src.utils.load_params import load_params

def evaluate_model(params: dict) -> dict:
    items = os.listdir(params["evaluation"]["models_dir"])
    numbers = [extract_number_from_string(item) for item in items if os.path.isdir(os.path.join(params["evaluation"]["models_dir"], item))]
    numbers = sorted([n for n in numbers if n is not None])
    best_model_fpath = Path(params["evaluation"]["models_dir"]) / f"train{numbers[-1]}" / "weights" / "best.pt"
    model = YOLO(best_model_fpath)

    res = model.predict(source="data/examples/001.jpg", conf=0.25)
    res = res[0].plot(line_width=1)
    res = res[:, :, ::-1]
    res = Image.fromarray(res)
    res.save("metrics/plots/result.jpg")


def extract_number_from_string(input_string):
    """
    Extract a number from a string using regular expressions.
    
    Args:
        input_string (str): The input string containing a number.
        
    Returns:
        int or None: Extracted number, or None if no number is found.
    """
    match = re.search(r'\d+', input_string)
    if match:
        return int(match.group())
    else:
        return None


if __name__ == "__main__":
    params = load_params("params.yaml")
    evaluate_model(params)
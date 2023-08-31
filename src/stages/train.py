import argparse
import logging

from src.utils.load_params import load_params
from src.utils.training_utilts import train_yolo

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
    params_path = args.config

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    params = load_params(params_path)
    train_yolo(params)
    
    
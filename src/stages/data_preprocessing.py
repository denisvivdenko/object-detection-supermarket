import argparse
import logging

from src.utils.load_params import load_params
from src.utils.data_preprocessing_utils import (
    load_dataset, 
    prepare_dataset
)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
    params_path = args.config

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    params = load_params(params_path)

    dataset = load_dataset(params)
    prepare_dataset(params, dataset)
    
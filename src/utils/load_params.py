import yaml


def load_params(fname: str) -> dict:
    with open(fname, 'r') as f:
        return yaml.safe_load(f)

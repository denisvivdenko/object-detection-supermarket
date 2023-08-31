from ultralytics import YOLO
import os
import shutil

def train_yolo(params: dict) -> None:
    model = YOLO(params["training"]["base_model_path"])
    model.train(
        data=params["training"]["yolo_params"],
        epochs=params["training"]["epochs"],
        batch=params["training"]["batch_size"],
        name=params["training"]["name"],
    )
    save_model(params)


def save_model(params: dict):
    """ saves the weights of trained model to the models directory """ 
    if os.path.isdir('runs'):
        model_weights = params["training"]["name"] + "/weights/best.pt"
        path_model_weights = os.path.join("runs/detect", model_weights)

        shutil.copy(src=path_model_weights, dst=params["training"]["save_fpath"])

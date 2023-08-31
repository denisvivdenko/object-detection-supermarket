from ultralytics import YOLO

def train_yolo(params: dict) -> None:
    model = YOLO(params["training"]["model_path"])
    model.train(
        data=params["training"]["yolo_params"],
        epochs=params["training"]["epochs"],
        batch=params["training"]["batch_size"]
    )
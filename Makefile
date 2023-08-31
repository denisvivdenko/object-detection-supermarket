deploy_data_processing_stage:
	dvc stage add --force --name data_preprocessing \
	--deps data/SKU110K_fixed \
	--outs data/dataset \
	--params base,yolo_preprocessing \
	python src/stages/data_preprocessing.py --config="params.yaml"

deploy_train_stage:
	dvc stage add --force --name train \
	--deps data/dataset \
	--deps models/yolov8n.pt \
	--outs models/yolo_trained_model.pt \
	--outs runs \
	--params base,yolo_preprocessing,training \
	python src/stages/train.py --config="params.yaml"

deploy_evaluate_stage:
	dvc stage add --force --name evaluate \
	--deps models/yolo_trained_model.pt \
	--plots metrics/plots \
	--params base,yolo_preprocessing,training,evaluation \
	python src/stages/evaluate.py --config="params.yaml"
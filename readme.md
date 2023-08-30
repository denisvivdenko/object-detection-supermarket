# Design Document

## Origin

    An application that enables you to collect all the data linked to your products and obtain all your key performance indicators in an instant.

    Our computer vision technology recognizes products and collects all visible data.

    This will give you a reliable and objective picture of the overall situation in shop.

### Goal
    1. Correctly detect SKUs on supermarket shelves (multiple object detection problem).
    2. Correctly classify SKUs from database.
    3. Run on mobile devices.
    4. Do not require internet access.
    
    This way a manager will be able to keep track of their shop from their phone and receive all the statistics needed.


## Modeling

    We will divide this task into to subsequent tasks

    Camera => Object detection model => Classification Model => statistics



### Object detection

#### Dataset 

https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd/edit

https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset

    We are going to use SKU110K dataset with YOLOv8 for multiple object detection problem

__TODO__
Explore how to implement it for mobile devices (without GPU)
- Quantization?

### SKU Classification model

- MobileNet: Developed by Google, MobileNet is optimized for mobile and embedded applications. It uses depthwise separable convolutions, which greatly reduces the number of parameters without compromising too much on performance.

- EfficientNet: Also from Google, EfficientNet uses a compound scaling method to scale the network depth, width, and resolution, which gives a good trade-off between accuracy and model size.

- SqueezeNet: Offers AlexNet-level accuracy with 50x fewer parameters.


## Why we divide this inference process into two models?

1. It is easier to debug
2. Dataset for training will be smaller
3. IMPORTANT! We will be easily adding new SKU when needed without need to retrain the while monolitic
solution, but just one image classification model.
4. We don't have dataset with for object detection with specified classes
5. Hard to augment data to train more robust model.

## Plan 

1. Finetune YOLOv8 model on SKU110K
2. Run this model on CPU
3. Collect dataset of any SKU (for example, Cocacola and Pepsi)
4. Augment data and finetune simple classification model to predict label
5. Infer through all the SKU110K dataset to collect statistics on volumes of pepsi vs cola


## MLOPS

The goal is to establish continuos integration pipleine that can be easily replicated and retrained
on examples that were wrongly classified or not detected.

We will use google drive to store dataset because it is easy to add new examples or check already
existing ones through interface.

For YOLO model we will establish an orchestration tool that will trigger the pipleine several times
per day to retrain the model and save the results to models repository.

The same thing we will do for classification model.

For inference we will create an endpoint on FastAPI that can be easily tested through localhost:8080/docs

We will use github actions and our custom runner for training process (Renting GPU on Vast.ai and 
running cml runner there)
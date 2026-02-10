### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation
effnetB2, effnetB2_transforms = create_effnetb2_model(
    num_classes=len(class_names)
)

# Load saved weights
effnetB2.load_state_dict(
    f=torch.load(f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"),
    map_location=torch.device('cpu')
)

### 3. Predict function ###
def predict(img) -> Tuple[Dict, float]:
    time = timer()
    transformed_img = effnetB2_transforms(img).unsqueeze(0).to("cpu")
    effnetB2_model.eval()
    with torch.no_grad():
        pred_probs = torch.softmax(effnetB2_model(transformed_img), dim=1)
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }
    return pred_labels_and_probs, round(timer() - time, 4)

### 4. Gradio app ###

# Create title, discription and article
title = "FoodVision Mini 🍕🥩🍣"
description = ("An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnetb2) computer vision model to classify images as pizza, steak or sushi.")
article = "Created at [09. PyTorch Model Deployment](https://github.com/ridamansour/PyTorchLearning/blob/main/09_pytorch_model_deployment.ipynb)"

# Create example list
example_list = [["examples/" + example] for example in os.listdir(foodvision_mini_examples_path)]

demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()

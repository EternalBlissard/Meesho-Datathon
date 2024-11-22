import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import random
from transformers import (
    ViTImageProcessor,
    ViTModel,
    ViTConfig,
    ViTPreTrainedModel,
    Trainer, 
    TrainingArguments,
    EfficientNetConfig, 
    EfficientNetModel,
    EfficientNetPreTrainedModel,
    TrainingArguments,
    AutoImageProcessor
    )
from sklearn.model_selection import train_test_split
import sys
from typing import List
from sklearn.metrics import classification_report,f1_score
import gc
import argparse
# from models import CustomConfig, MultiLabelMultiClassViT
import wandb
import os

class CustomConfig(EfficientNetConfig):
    def __init__(self,num_classes_per_label:List[int]=[1],**kwargs):
        super().__init__(**kwargs)
        self.num_classes_per_label = num_classes_per_label

class MultiLabelMultiClassEff(EfficientNetPreTrainedModel):
    config_class=CustomConfig
    def __init__(self, config: CustomConfig) -> None:
        super().__init__(config)

        # self.vit = ViTModel(config, add_pooling_layer=True  )
        self.efficientnet = EfficientNetModel(config)
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Dropout(0.2),
            # nn.Linear(config.hidden_size, 32),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(49, num_classes)) 
            for num_classes in config.num_classes_per_label
        ])
        # self.weights = torch.tensor([1,10],dtype=torch.float32).to(DEVICE)
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self, pixel_values,labels=None):
        # print(pixel_values.shape)   
        outputs = self.efficientnet(pixel_values).last_hidden_state[:, 0, :]  # CLS token representation
        outputs = outputs.reshape(outputs.shape[0],-1)
        # print(outputs.shape)
        logits = [classifier(outputs) for classifier in self.classifiers]
        
        if labels is not None:
            loss=0
            for i in range(len(logits)):
                target=labels[:,i]
                loss += torch.nn.functional.cross_entropy(logits[i], target)#, weight=self.weights)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}



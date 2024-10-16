import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
from transformers import ViTModel
import warnings
warnings.filterwarnings("ignore")
import random
import time
from transformers import ViTImageProcessor
import sys
from typing import List
from transformers import ViTConfig,ViTPreTrainedModel
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report
import gc

DEVICE="cuda:0"
def setAllSeeds(seed):
  os.environ['MY_GLOBAL_SEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
setAllSeeds(42)

df = pd.read_csv("train.csv")
categories=df["Category"].unique()
print(categories)
category=categories[4]
df = df[df["Category"]==category]
test_df = pd.read_csv("test.csv")
test_df = test_df[test_df["Category"]==category]
test_df.head()

id2label={}
label2id={}
attrs={}
total_attr=len(df.columns)
for i in range(3,total_attr):
    labels=df[df.columns[i]].unique()
    id2label[i-3]={k:labels[k] for k in range(len(labels))}
    label2id[i-3]={labels[k]:k for k in range(len(labels))}
    attrs[i-3]=df.columns[i]
print(id2label)
print(label2id)
print(attrs)

model_name = f'vit/{category}/final'
processor = ViTImageProcessor.from_pretrained(model_name)

class CustomFashionManager(Dataset):
    def __init__(self,csv_file, root_dir="./",transforms =None):
        self.fashionItems = csv_file
        self.root_dir = root_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.fashionItems)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,f"{self.fashionItems.iloc[idx, 0]:06d}"+'.jpg')
        image = Image.open(img_name)
        inputs=processor(image, return_tensors='pt')
        # if self.transforms:
        #     sample = self.transforms(sample)
        return inputs
    
train_fashion_data = CustomFashionManager(csv_file=df,root_dir='train_images')
test_fashion_data = CustomFashionManager(csv_file=test_df,root_dir='test_images')


class CustomConfig(ViTConfig):
    def __init__(self,num_classes_per_label:List[int]=[1],**kwargs):
        super().__init__(**kwargs)
        self.num_classes_per_label = num_classes_per_label

class MultiLabelMultiClassViT(ViTPreTrainedModel):
    config_class=CustomConfig
    def __init__(self, config: CustomConfig) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False)
        self.classifiers = nn.ModuleList([
            nn.Linear(config.hidden_size, num_classes) 
            for num_classes in config.num_classes_per_label
        ])
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, pixel_values,labels=None):
        outputs = self.vit(pixel_values).last_hidden_state[:, 0, :]  # CLS token representation
        logits = [classifier(outputs) for classifier in self.classifiers]
        if labels is not None:
            loss=0
            for i in range(len(logits)):
                target=labels[:,i]
                loss += torch.nn.functional.cross_entropy(logits[i], target)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

model = MultiLabelMultiClassViT.from_pretrained(model_name)


def collate_fn(batch):
    return {
        'pixel_values': torch.cat([x['pixel_values'] for x in batch],dim=0),
    }

def compute_metrics(pred):
    logits = pred.predictions
    labels=pred.label_ids
    probs = np.stack([np.argmax(logit,axis=1) for logit in logits])
    probs=probs.T
    report=classification_report(labels.flatten(),probs.flatten(),output_dict=True)
    return {'accuracy': report['accuracy'],"macro avg f1":report['macro avg']['f1-score']}

training_args = TrainingArguments(
  output_dir="./vit/"+category,
  per_device_train_batch_size=96,
  per_device_eval_batch_size=96,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  num_train_epochs=1,
  fp16=True,
  learning_rate=2e-4,
  save_total_limit=1,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_fashion_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

y_pred=trainer.predict(test_fashion_data)
logits = y_pred.predictions
probs = np.stack([np.argmax(logit,axis=1) for logit in logits])
probs=probs.T
l=[]
for i in range(len(probs)):
    x=[]
    for j in range(len(probs[i])):
        x.append(id2label[j][probs[i][j]])
    l.append(x)
test_df['len']=len(l[0])
for i in range(10):
    x=[]
    for j in range(len(l)):
        if i<len(l[0]) and l[j][i]!=np.nan:
            x.append(l[j][i])
        else:
            x.append(np.nan)
    test_df[f"attr_{i+1}"]=x

test_df.to_csv(f"preds/{category}.csv",index=False)

del model, trainer
torch.cuda.empty_cache()
gc.collect()



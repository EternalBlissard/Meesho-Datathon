import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    )
from sklearn.model_selection import train_test_split
import sys
from typing import List
from sklearn.metrics import classification_report
import gc
import argparse


parser = argparse.ArgumentParser(description="category parser")
parser.add_argument("-ci","--category_idx", type=int, default=0,choices=[0,1,2,3,4],help="category index")
args = parser.parse_args()
model_name = 'google/vit-base-patch16-224'
save_dir="./vit3/"
DEVICE="cuda:1"
def setAllSeeds(seed):
  os.environ['MY_GLOBAL_SEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
setAllSeeds(42)


df = pd.read_csv("train.csv")
categories=df["Category"].unique()

category=categories[args.category_idx]
df = df[df["Category"]==category]
save_dir+=category

delCol = []
trackNum = []
for i in range(1,11):
    uniName = df["attr_"+str(i)].unique()
    if(len(uniName)==1):
        delCol.append("attr_"+str(i))
    else:
        trackNum.append(len(uniName))

df = df.drop(delCol,axis=1)
# df = df.fillna("NA")

id2label={}
label2id={}
attrs={}
total_attr=len(df.columns)
for i in range(3,total_attr):
    labels=df[df.columns[i]].dropna().unique()
    id2label[i-3]={k:labels[k] for k in range(len(labels))}
    label2id[i-3]={labels[k]:k for k in range(len(labels))}
    attrs[i-3]=df.columns[i]
print(id2label)
print(label2id)
print(attrs)

def categorize(example):
    for i in attrs:
        # print(example[attrs[i]],type(example[attrs[i]]),pd.isna(example[attrs[i]]))
        if not pd.isna(example[attrs[i]]):
            example[attrs[i]]=label2id[i][example[attrs[i]]]
        else:
            example[attrs[i]]=-100
    return example
df=df.apply(categorize,axis=1)

processor = ViTImageProcessor.from_pretrained(model_name)

#train test split
# train_df, val_df = train_test_split(df, test_size=0.3)
# val_df,test_df=train_test_split(val_df,test_size=0.33)
train_df=df
val_df=df
test_df=df


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
        attributes = self.fashionItems.iloc[idx, 3:]
        attributes = np.array(attributes)
        attributes = attributes.astype('float')
        inputs=processor(image, return_tensors='pt')
        # if self.transforms:
        #     inputs = self.transforms(inputs)
        inputs['labels']=torch.tensor(attributes, dtype=torch.long)
        return inputs

train_fashion_data = CustomFashionManager(csv_file=train_df,root_dir='train_images')
val_fashion_data = CustomFashionManager(csv_file=val_df,root_dir='train_images')
test_fashion_data = CustomFashionManager(csv_file=test_df,root_dir='train_images')

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

def collate_fn(batch):
    return {
        'pixel_values': torch.cat([x['pixel_values'] for x in batch],dim=0),
        'labels': torch.stack([x['labels'] for x in batch])
    }

def compute_metrics(pred):
    logits = pred.predictions
    labels=pred.label_ids
    probs = np.stack([np.argmax(logit,axis=1) for logit in logits])
    probs=probs.T
    labels=labels.flatten()
    probs=probs.flatten()
    non_padding_indices = [i for i, label in enumerate(labels) if label != -100]
    labels = [labels[i] for i in non_padding_indices]
    probs = [probs[i] for i in non_padding_indices]
    report=classification_report(labels,probs,output_dict=True)
    return {'accuracy': report['accuracy'],"macro avg f1":report['macro avg']['f1-score']}

config=ViTConfig.from_pretrained(model_name)
config=CustomConfig(num_classes_per_label=trackNum,**config.to_dict())
model = MultiLabelMultiClassViT.from_pretrained(model_name,config=config)

training_args = TrainingArguments(
  output_dir="./vit3/"+category,
  per_device_train_batch_size=128,
  per_device_eval_batch_size=128,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_strategy="epoch",
  num_train_epochs=5,
  fp16=True,
  learning_rate=2e-4,
  save_total_limit=1,
  remove_unused_columns=False,
  report_to='wandb',
  load_best_model_at_end=True,
  metric_for_best_model="macro avg f1"
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_fashion_data,
    eval_dataset=val_fashion_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
trainer.train()
trainer.save_model(f"./vit3/{category}/final")
# trainer.evaluate(test_fashion_data)

del model, trainer
torch.cuda.empty_cache()
gc.collect()



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from modelsEff import CustomConfig, MultiLabelMultiClassEff
import wandb
import os

root_dir='/notebooks/Meesho-Datathon/training/visual-taxonomy/train_images'
train_csv = "/notebooks/Meesho-Datathon/training/train.csv"
#categories=['Men Tshirts' 'Sarees' 'Kurtis' 'Women Tshirts' 'Women Tops & Tunics']
parser = argparse.ArgumentParser(description="category parser")
parser.add_argument("-ci","--category_idx", type=int, default=3,choices=[0,1,2,3,4],help="category index")
args = parser.parse_args()


model_name = 'google/efficientnet-b0'
save_dir="../models/"
DEVICE="cuda:0"
def setAllSeeds(seed):
  os.environ['MY_GLOBAL_SEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
# setAllSeeds(20)


df = pd.read_csv(train_csv)
categories=df["Category"].unique()

category=categories[args.category_idx]
df = df[df["Category"]==category]
# save_dir+=category

attr_i=8
removing_attri=[]
df.iloc[:,3:2+attr_i]=np.nan
df.iloc[:,3+attr_i:]=np.nan

wandb_api_key = os.getenv("WANDB_API")
wandb.login(key=wandb_api_key)
run = wandb.init(
    project='huggingface', 
    job_type="training", 
    anonymous="allow",
    name=category,
)

delCol = []
trackNum = []
for i in range(1,11):
    uniName = df["attr_"+str(i)].dropna().unique()
    if(len(uniName)==0):
        delCol.append("attr_"+str(i))
    else:
        trackNum.append(len(uniName))

df = df.drop(delCol,axis=1)
if(len(removing_attri)==0 and attr_i>0):
    df = df.dropna()


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

processor = AutoImageProcessor.from_pretrained(model_name)


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# test_df=df


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
        inputs['labels']=torch.tensor(attributes, dtype=torch.long)
        return inputs

train_fashion_data = CustomFashionManager(csv_file=train_df,root_dir=root_dir)
val_fashion_data = CustomFashionManager(csv_file=val_df,root_dir=root_dir)


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
    macros = []
    micros = []
    f1s=[]
    for i in range(labels.shape[1]):
        non_padding_indices = [j for j, label in enumerate(labels[:,i]) if label != -100]
        labels_ = [labels[j,i] for j in non_padding_indices]
        probs_ = [probs[j,i] for j in non_padding_indices]
        micro=f1_score(labels_,probs_,average='micro')
        macro=f1_score(labels_,probs_,average='macro')
        print(f"attr_{i+1} macro_f1 score: {macro}")
        print(f"attr_{i+1} micro_f1 score: {micro}")
        # print(classification_report(labels_,probs_))
        score=2*(micro*macro)/(micro+macro)
        print(f"attr_{i+1} score: {score}")
        f1s.append(score)
        macros.append(macro)
        micros.append(micro)
    
    wandb.log({'score': sum(f1s)/len(f1s)})
    return {'score': sum(f1s)/len(f1s), 'macro_f1':sum(macros)/len(macros),'micro_f1':sum(micros)/len(micros)}

config=EfficientNetConfig.from_pretrained(model_name)
config=CustomConfig(num_classes_per_label=trackNum,**config.to_dict())
model = MultiLabelMultiClassEff.from_pretrained(model_name,config=config)

training_args = TrainingArguments(
  output_dir=save_dir+category+"_EffAttr8",
  per_device_train_batch_size=128,
  per_device_eval_batch_size=128,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_strategy="epoch",
  num_train_epochs=10,
  fp16=True,
  learning_rate=2e-4,
  save_total_limit=1,
  remove_unused_columns=False,
  report_to='wandb',
  load_best_model_at_end=True,
  metric_for_best_model="score"
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
trainer.save_model(f"{save_dir}{category}_EffAttr8/final")
# trainer.evaluate(test_fashion_data)

del model, trainer
torch.cuda.empty_cache()
gc.collect()



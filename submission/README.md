#HOW TO RUN

#REQUIREMENTS
```
pip install -r -U requirements.txt
```
Must have a <a href='https://wandb.ai'></a>wandb ID to save logs and run the code

#TRAINING
For Each File in ./training alter the values of line 33 and 34 <br>

```
root_dir= path_to_train_images 
train_csv = path_to_train.csv
```

Run the Code
```
python {Unique_Identifier}_train.py
```
#INFERENCE

For Each File in ./inference

#DOWNLOAD THE MODELS FROM <a href='https://drive.google.com/drive/folders/1baj-9lZhaczh84H0AEkHciz2cZL6yRGI?usp=sharing'>Here</a> UNZIP AND PLACE THE FOLDER WITHIN THE SUBMISSIONS FOLDER ELSE ALTER THE FILE PATH on LINE 67
```
model_name = model_path
```

DO CHANGE THE FOLLOWING PATHS ON LINES 29-31
```
train_csv = path_to_train.csv
test_csv  = path_to_test.csv
root_dir= path_to_folder_having_train_images_and_test_images
```

Run the Code in 
```
python {Unique_Identifier}_infer.py
```

#Generate_Submission

```
python combine_preds.py
```



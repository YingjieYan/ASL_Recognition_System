from clearml import Task, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

task = Task.init(project_name="ASL_Classification", task_name="Pipeline Step 2: Preprocess Data")

# task.execute_remotely()

# get parameters
args = Task.current_task().get_parameters_as_dict()
dataset_task_id = args.get("General/dataset_task_id", None)
if not dataset_task_id:
    dataset_task_id = '541a4f0f5e5948548e781553d7e9366a'  # 替换为你实际的 dataset 上传任务 ID

task.connect({"General/dataset_task_id": dataset_task_id})

# get the last step of metadata artifact
meta_task = Task.get_task(task_id=dataset_task_id)
df = meta_task.artifacts['asl_metadata_df'].get()

# get the dataset path
dataset = Dataset.get(dataset_project="ASL_Classification", dataset_name="ASL_Images")
img_root_path = dataset.get_local_copy()

# get the part of the data
df_sample = df.sample(frac=0.1, random_state=42)
train_df = df_sample.sample(frac=0.8, random_state=42)
test_df = df_sample.drop(train_df.index)

# adjust image size
IMG_SIZE = (64, 64)

def load_images_labels(df, root_path):
    X = []
    y = []
    label_map = {label: idx for idx, label in enumerate(sorted(df['class'].unique()))}
    for _, row in df.iterrows():
        img_path = os.path.join(root_path, row['filename'])
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, IMG_SIZE)
            X.append(image)
            y.append(label_map[row['class']])
    return np.array(X), np.array(y)

X_train, y_train = load_images_labels(train_df, img_root_path)
X_test, y_test = load_images_labels(test_df, img_root_path)

# upload data and dataframe
task.upload_artifact('train_df', train_df)
task.upload_artifact('test_df', test_df)
task.upload_artifact('X_train', X_train)
task.upload_artifact('y_train', y_train)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_test', y_test)

print("Step 2 done. Uploaded train/test splits and image arrays.")

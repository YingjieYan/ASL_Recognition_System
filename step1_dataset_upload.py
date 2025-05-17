from clearml import Task, Dataset
import pandas as pd
import os

# Init ClearML Task
task = Task.init(project_name="ASL_Classification", task_name="Pipeline Step 1: Upload Dataset")

# task.execute_remotely()

# get dataset by name
dataset = Dataset.get(dataset_project="ASL_Classification", dataset_name="ASL_Images")

# download and extract dataset
local_path = dataset.get_local_copy()

# create DataFrame
data = []
for class_name in os.listdir(local_path):
    class_folder = os.path.join(local_path, class_name)
    if os.path.isdir(class_folder):
        files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            rel_path = os.path.join(class_name, f)
            data.append({"filename": rel_path, "class": class_name})

df = pd.DataFrame(data)

# upload dataframe for artifact
task.upload_artifact(name='asl_metadata_df', artifact_object=df)

print("Step 1 done. Uploaded asl_metadata_df artifact with image metadata.")

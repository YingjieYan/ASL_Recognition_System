# upload_dataset.py
from clearml import Dataset

# change to your local_path
dataset_path = 'ASL_Dataset/asl_alphabet_train/asl_alphabet_train/'

dataset = Dataset.create(
    dataset_name="ASL_Images",
    dataset_project="ASL_Classification"
)

dataset.add_files(path=dataset_path)
dataset.upload()
dataset.finalize()

print("Dataset uploaded successfully")

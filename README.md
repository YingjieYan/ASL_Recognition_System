
# ASL Alphabet Image Classification with ClearML Pipeline
This project uses ClearML to implement a full ASL (American Sign Language) alphabet image classification workflow using deep learning and computer vision.

## 🎯Objective
The product sign language recognition system can be used in hospitals for doctors to understand the meaning of Deaf 
mute. It also can be an assistant tool to help normal people to learn sign language quickly. Besides that, some smart 
devices can detect sign language and perform the corresponding operation. By integrating advanced computer vision and 
deep learning techniques, our system recognizes and interprets hand gestures. Expand our organization’s reach into the 
healthcare, education, and smart technology markets.
## 🚀 Project structure
```text
ASL/
├── step1_upload_dataset.py      # Upload image dataset and generate metadata
├── step2_preprocessing.py       # Load and preprocess images, upload training/test sets
├── step3_train_model.py         # Train the CNN model and save the weights
├── main.py                      # ClearML Pipeline controller
├── upload_dataset.py            # Upload local data
└── README.md
text```
# Getting Started
## 1. Install Dependencies
  pip install clearml
  pip install clearml-agent
## 2. Configure ClearML
Paste the credential into the clearML workspace
  clearml-init
## 3. Upload local datasets to clearML datasets
  python upload_dataset.py

# Run three steps and store it in the ASL_Classification project
## 1 Upload image dataset and generate metadata
  python step1_dataset_upload.py
## 2 Load and preprocess images, upload training/test sets
  python step2_preprocessing.py
## 3 Train the CNN model and save the weights
  python step3_train_model.py
## 4 start ClearML Agent
  clearml-agent daemon --queue pipeline --detached
## 5 Run the pipeline controller to register its three steps into ASL_Pipeline
  python main.py
# The following is the pipeline operation diagram

<img width="293" alt="readme" src="https://github.com/user-attachments/assets/a003b172-2e23-4041-95c2-804cfe1ee946" />


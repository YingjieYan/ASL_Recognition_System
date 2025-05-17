
# ASL Alphabet Image Classification with ClearML Pipeline
This project uses ClearML to implement a full ASL (American Sign Language) alphabet image classification workflow using deep learning and computer vision.

## 🎯Objective
The product sign language recognition system can be used in hospitals for doctors to understand the meaning of Deaf 
mute. It also can be an assistant tool to help normal people to learn sign language quickly. Besides that, some smart 
devices can detect sign language and perform the corresponding operation. By integrating advanced computer vision and 
deep learning techniques, our system recognizes and interprets hand gestures. Expand our organization’s reach into the 
healthcare, education, and smart technology markets.
## 🚀 Project structure
<pre>
ASL/
├── step1_upload_dataset.py      # Upload image dataset and generate metadata
├── step2_preprocessing.py       # Load and preprocess images, upload training/test sets
├── step3_train_model.py         # Train the CNN model and save the weights
├── main.py                      # ClearML Pipeline controller
├── upload_dataset.py            # Upload local data
└── README.md
</pre>
# Getting Started
## 1. Install Dependencies
<pre>
  pip install clearml
  pip install clearml-agent
</pre>
## 2. Configure ClearML
Create a credential from the clearml workspace and paste it above
<pre>
  clearml-init
</pre>
## 3. Upload local datasets to clearML datasets
<pre>
  python upload_dataset.py
</pre>
# Run three steps and store it in the ASL_Classification project
Before starting the following steps, you need to create a new queue called pipeline in the works & queues of clearml, so that subsequent agents can listen to the queue and run the project steps according to their pipeline order.
## 1 Upload image dataset and generate metadata
 <pre> python step1_dataset_upload.py</pre>
## 2 Load and preprocess images, upload training/test sets
  <pre> python step2_preprocessing.py</pre>
## 3 Train the CNN model and save the weights
   <pre>  python step3_train_model.py  </pre> 
## 4 start ClearML Agent
  <pre> clearml-agent daemon --queue pipeline --detached  </pre> 
## 5 Run the pipeline controller to register its three steps into ASL_Pipeline
   <pre>  python main.py  </pre> 
# The following is the pipeline operation diagram

<img width="293" alt="readme" src="https://github.com/user-attachments/assets/a003b172-2e23-4041-95c2-804cfe1ee946" />


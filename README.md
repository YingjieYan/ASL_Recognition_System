
# Real-Time ASL Sign Language Detection
This project uses ClearML to implement a full ASL (American Sign Language) alphabet image classification workflow using deep learning and computer vision.

## 🎯Objective
This project is dedicated to developing a system capable of real-time recognition of American Sign Language (ASL) letters, aiming to support individuals with hearing impairments in achieving smoother communication in everyday life. The system captures users' hand movements via a webcam and employs deep learning models to classify ASL letters, displaying the recognised results as on-screen captions.

In educational settings, the system serves as a visual and interactive training platform for ASL learners, enhancing their learning efficiency. In public spaces such as transport hubs, service centres, and self-service kiosks, it can assist staff in communicating basic information with deaf or hard-of-hearing individuals. For daily interactions, the system provides real-time sign-to-text conversion, along with a speech function that vocalises the recognised content, improving convenience and natural communication.

In addition, the system supports features such as dynamic caption assembly, text-to-speech output, and content clearing, offering a user-friendly interface and strong extensibility. It is well-suited to a variety of human–computer interaction scenarios, contributing to a more inclusive communication environment.


## 🚀 Project structure
<pre>
ASL/
├── .github/workflows/  # CI/CD automation workflow
│
├── app.py # Flask-based GUI for real-time ASL recognition
├── index.html # HTML interface for front-end UI
├── script.js # JavaScript logic for front-end interaction
│
├── step1_load_landmark_dataset_from_clearml_dataset.py # Step 1: Load dataset from ClearML
├── step2_split_loaded_data.py # Step 2: Split dataset into train/test sets
├── step3_train_evaluate_landmark_mlp.py # Step 3: Train & evaluate model
├── upload_dataset_for_landmarks.py # Upload local dataset to ClearML
├── main.py # Core pipeline runner that links ClearML steps
├── hpo.py # Hyperparameter optimization using ClearML + Optuna
│
├── trained_asl_landmark_mlp_local.keras # Trained MLP model
│
├── requirements.txt # Project dependencies
├── README.md # Project overview and usage guide
</pre>
## Getting Started
### 1. Install Dependencies
<pre>
  pip install clearml
  pip install clearml-agent
</pre>
### 2. Configure ClearML
Create a credential from the clearml workspace and paste it above
<pre>
  clearml-init
</pre>
### 3. Upload local datasets to clearML datasets
<pre>
  python upload_dataset_for_landmarks.py
</pre>
## Run three steps and store it in the ASL_Classification project
Before starting the following steps, you need to create a new queue called pipeline in the works & queues of clearml, so that subsequent agents can listen to the queue and run the project steps according to their pipeline order.
### 1. Upload image dataset and generate metadata
 <pre> python step1_load_landmark_dataset_from_clearml_dataset.py</pre>
### 2. Load and preprocess images, upload training/test sets
  <pre> python step2_split_loaded_data.py</pre>
### 3. Train model and save the weights
   <pre>  python step3_train_evaluate_landmark_mlp.py  </pre> 
### 4. start ClearML Agent
  <pre> clearml-agent daemon --queue pipeline --detached  </pre> 
### 5. Run the pipeline controller to register its three steps into ASL_Pipeline
   <pre>  python main.py  </pre> 

## 📊 ClearML Pipeline Overview

<img width="293" alt="readme" src="https://github.com/user-attachments/assets/a003b172-2e23-4041-95c2-804cfe1ee946" />

## 🧪 Hyperparameter Optimisation (HPO)

The `hpo.py` script performs automated hyperparameter tuning using **ClearML** in combination with **Optuna**.

### 🔍 Purpose
This module aims to identify the optimal set of hyperparameters for the MLP model used in ASL landmark classification. It leverages Optuna's efficient sampling strategies and ClearML's experiment tracking to perform and visualise multiple trials.

### 🛠️ Key Features
- Integrated with ClearML’s `HyperParameterOptimizer` engine.
- Automatically logs and compares trial results in the ClearML dashboard.
- Supports customisable search spaces for parameters:
  - Learning rate
  - Batch size
  - Dropout rate

### 🚀 How to Run
```bash
python hpo.py
```

## 🖥️ User Interface

We developed a clean and responsive **web-based user interface using Flask**, implemented in `app.py`.  
This interface enables users to interact with the ASL recognition system in real time.
### 🖼️ Demo Interface
![GUI](https://github.com/YingjieYan/ASL_Recognition_System/raw/main/GUI.jpg)

### Key Features:
- 📷 **Live webcam integration**: Capture hand gestures directly from your camera  
- 🔤 **Real-time ASL letter recognition**: View model predictions instantly as subtitles  
- 🔊 **Text-to-speech support**: Convert recognised text to speech for easier communication  
- 🧹 **Clear content**: One-click to reset recognised output  
- 💬 **Dynamic caption assembly**: Accumulate multiple letters into words or sentences

### 🚀 How to Launch

To start the web interface locally, run:
```bash
python app.py
```
Visit the address shown in the terminal (usually http://127.0.0.1:5000) to access the app.


## ⚙️ CI/CD Pipeline

This project integrates a **CI/CD workflow using GitHub Actions** to automate the execution of the ClearML pipeline. Every time code is pushed to the `CI/CD` branch, the pipeline is automatically triggered to ensure seamless model updates and reproducibility.

**Workflow features:**
- 🔁 Automatically runs when pushing to the `CI/CD` branch.
- 🐍 Sets up a Python 3.12 environment.
- 📦 Installs dependencies from `requirements.txt`.
- 🚀 Submits the ClearML pipeline via `main.py`.
- 🔐 Uses GitHub Secrets to securely manage ClearML API credentials.

**CI/CD file:** `.github/workflows/pipeline.yml`

## 📬 We Value Your Feedback!
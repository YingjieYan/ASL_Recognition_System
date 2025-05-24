
# ASL Alphabet Image Classification with ClearML Pipeline
This project uses ClearML to implement a full ASL (American Sign Language) alphabet image classification workflow using deep learning and computer vision.

## 🎯Objective
The Sign Language Recognition System is designed to bridge communication gaps and promote inclusivity across multiple sectors. In healthcare, it enables doctors and medical staff to understand the needs of Deaf-mute patients through gesture interpretation. In education, it serves as an assistive tool to help individuals learn sign language more efficiently. Additionally, the system can be integrated into smart devices to enable gesture-based control and interaction.

By leveraging advanced computer vision and deep learning techniques, our solution accurately recognises and interprets hand gestures in real time. This technology empowers our organisation to expand into the healthcare, education, and smart technology markets with an accessible and intelligent communication solution.

## 🚀 Project structure
<pre>
ASL/
├── .github/workflows/             # CI/CD workflows (e.g., model training or deployment automation via GitHub Actions)
├── app.py                         # Streamlit-based GUI for real-time ASL recognition and user feedback
├── ASL.py                         # Main logic for handling ASL recognition using MediaPipe and MLP
├── ASL_CNN.py                     # Legacy CNN-based recognition script (for reference or comparison)
├── asl_cnn_model.h5               # Pretrained CNN model weights (legacy model)
├── main.py                        # ClearML pipeline controller that links all pipeline steps
├── step1_dataset_upload.py        # Step 1: Upload raw dataset and create metadata on ClearML
├── step2_preprocess.py            # Step 2: Extract and normalise MediaPipe landmarks for model training
├── step3_train_model.py           # Step 3: Train MLP model on preprocessed landmark data
├── upload_dataset.py              # Utility script for manually uploading local datasets
├── requirements.txt               # Python dependencies for environment setup
├── README.md                      # Project overview, setup guide, and usage instructions
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
## 1. Upload image dataset and generate metadata
 <pre> python step1_dataset_upload.py</pre>
## 2. Load and preprocess images, upload training/test sets
  <pre> python step2_preprocessing.py</pre>
## 3. Train model and save the weights
   <pre>  python step3_train_model.py  </pre> 
## 4. start ClearML Agent
  <pre> clearml-agent daemon --queue pipeline --detached  </pre> 
## 5. Run the pipeline controller to register its three steps into ASL_Pipeline
   <pre>  python main.py  </pre> 

# 📊 ClearML Pipeline Overview

<img width="293" alt="readme" src="https://github.com/user-attachments/assets/a003b172-2e23-4041-95c2-804cfe1ee946" />

# ⚙️ CI/CD Pipeline

This project integrates a **CI/CD workflow using GitHub Actions** to automate the execution of the ClearML pipeline. Every time code is pushed to the `CI/CD` branch, the pipeline is automatically triggered to ensure seamless model updates and reproducibility.

**Workflow features:**
- 🔁 Automatically runs when pushing to the `CI/CD` branch.
- 🐍 Sets up a Python 3.12 environment.
- 📦 Installs dependencies from `requirements.txt`.
- 🚀 Submits the ClearML pipeline via `main.py`.
- 🔐 Uses GitHub Secrets to securely manage ClearML API credentials.

**CI/CD file:** `.github/workflows/pipeline.yml`

# 🖥️ User Interface

We built an intuitive and interactive **web-based user interface using Streamlit**, implemented in `app.py`.  
This interface allows users to:

- 📸 Capture live input or upload ASL gesture images  
- 🤖 View real-time prediction results from the trained model  
- 📝 Submit feedback on incorrect predictions to support future improvements  
- 🔄 Seamlessly test the system with no additional setup

To launch the UI locally, simply run:

```bash
streamlit run app.py
```

## 📬 We Value Your Feedback!
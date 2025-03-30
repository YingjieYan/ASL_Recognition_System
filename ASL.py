import os # import operating system module
import shutil # import file operation module
import random # import random number module
from sklearn.model_selection import train_test_split # Import dataset split function from sklearn library

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Define the root directory of the training set
train_dir = '/ASL_Dataset/asl_alphabet_train/asl_alphabet_train/'

# Traverse each class folder and build the file list and corresponding labels
data = []
# Assume that each subfolder in train_dir is named after the class
for class_name in os.listdir(train_dir):
    class_folder = os.path.join(train_dir, class_name)
    if os.path.isdir(class_folder):
        # Get all image files (you can adjust the file suffixes as needed)
        files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            # Record relative path for later use with flow_from_dataframe
            rel_path = os.path.join(class_name, f)
            data.append({"filename": rel_path, "class": class_name})

# Create a DataFrame
df = pd.DataFrame(data)
print("Total number of samples:", len(df))

# Randomly sample 10% of the dataset (random_state ensures reproducibility)
df_sample = df.sample(frac=0.1, random_state=42)
print("Number of samples after sampling:", len(df_sample))

# Split the sampled data into training and test sets (80/20 split)
train_df = df_sample.sample(frac=0.8, random_state=42)
test_df = df_sample.drop(train_df.index)
print("Number of training samples:", len(train_df))
print("Number of test samples:", len(test_df))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Brightness-enhanced training data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=(0.9, 1.3)
)

# Normalization-only for testing
test_datagen = ImageDataGenerator(rescale=1./255)
# Create training data generator using flow_from_dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,  # Root directory for relative paths
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=30,
    shuffle=True,
    class_mode='categorical'
)

# Create testing data generator using flow_from_dataframe
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=train_dir,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=30,
    shuffle=False,
    class_mode='categorical'
)

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# Define the CNN model
model_CNN = Sequential()
model_CNN.add(InputLayer(input_shape=(224, 224, 3)))

model_CNN.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Flatten())
model_CNN.add(Dense(2048, activation='relu'))
model_CNN.add(Dropout(0.25))

model_CNN.add(Dense(29, activation='softmax'))

# Model overview
model_CNN.summary()

# Compile the model
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
model_CNN.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 128 # defines the number of samples
history_CNN = model_CNN.fit(
    train_generator,
    epochs=10, # defines the number of epochs for model training
    validation_data=test_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=test_generator.samples // batch_size
)
# save model
save_path = "model_CNN.h5"
model_CNN.save(save_path)

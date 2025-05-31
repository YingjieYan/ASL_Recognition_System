# upload_dataset_for_landmarks.py
from clearml import Dataset
import os
import cv2
import mediapipe as mp
import numpy as np

# --- Configuration ---
# Path to the local ASL alphabet image dataset (update this as needed)
LOCAL_RAW_IMAGE_DATASET_ROOT_PATH = 'ASL_Dataset/asl_alphabet_train/asl_alphabet_train/' 

# ClearML Dataset Configuration
CLEARML_DATASET_PROJECT = "ASL_Classification_Pipeline" 
CLEARML_LANDMARK_DATASET_NAME = "ASL_Landmark_Features_NPZ" 

# Landmark Generation Configuration
USE_3D_LANDMARKS = True
# --- End Configuration ---

mp_hands = mp.solutions.hands

def normalize_landmarks(landmarks_world_coords):

    if landmarks_world_coords is None or len(landmarks_world_coords) == 0:
        return None
    wrist_coords = landmarks_world_coords[0].copy()
    normalized_coords = landmarks_world_coords - wrist_coords
    palm_size = np.linalg.norm(normalized_coords[9])
    if palm_size < 1e-6:
        return None
    normalized_coords = normalized_coords / palm_size
    return normalized_coords.flatten()

def main():
    if not os.path.isdir(LOCAL_RAW_IMAGE_DATASET_ROOT_PATH):
        print(f"Error: Raw image dataset path '{LOCAL_RAW_IMAGE_DATASET_ROOT_PATH}' not found.")
        return

    print(f"--- Generating Landmark Features from Raw Images ---")
    all_landmarks_data = []
    all_labels_str = []

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:

        for class_name in sorted(os.listdir(LOCAL_RAW_IMAGE_DATASET_ROOT_PATH)):
            class_folder = os.path.join(LOCAL_RAW_IMAGE_DATASET_ROOT_PATH, class_name)
            if not os.path.isdir(class_folder):
                continue
            print(f"Processing class: {class_name}")

            image_files_in_class = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files_in_class:
                print(f"  No image files found in {class_folder}")
                continue
            
            processed_count = 0
            for image_name in image_files_in_class:
                image_path = os.path.join(class_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  Warning: Could not read image {image_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks_mp = results.multi_hand_landmarks[0]
                    landmark_coords_list = []
                    for landmark in hand_landmarks_mp.landmark:
                        if USE_3D_LANDMARKS:
                            landmark_coords_list.append([landmark.x, landmark.y, landmark.z])
                        else:
                            landmark_coords_list.append([landmark.x, landmark.y])
                    
                    landmark_coords_np = np.array(landmark_coords_list)
                    normalized_flat_landmarks = normalize_landmarks(landmark_coords_np)
                    
                    if normalized_flat_landmarks is not None:
                        all_landmarks_data.append(normalized_flat_landmarks)
                        all_labels_str.append(class_name)
                        processed_count += 1
            print(f"  Processed {processed_count} images for class {class_name}")

    if not all_landmarks_data:
        print("Error: No landmark data was extracted. Aborting ClearML Dataset creation.")
        return

    X_data = np.array(all_landmarks_data)
    unique_labels_sorted = sorted(list(set(all_labels_str)))
    label_to_id_map = {label: i for i, label in enumerate(unique_labels_sorted)}
    y_numeric_data = np.array([label_to_id_map[label] for label in all_labels_str])

    print(f"--- Landmark Feature Generation Complete ---")
    print(f"Extracted landmarks: X_data shape {X_data.shape}, y_numeric_data shape {y_numeric_data.shape}")
    print(f"Generated label map: {label_to_id_map}")

    # --- Uploading Processed Landmark Data to ClearML Dataset ---
    print(f"\nCreating ClearML Dataset '{CLEARML_LANDMARK_DATASET_NAME}' in project '{CLEARML_DATASET_PROJECT}' for landmark features...")
    # Get the dataset object (or create a new version if it exists)
    try:
        dataset = Dataset.get(
            dataset_project=CLEARML_DATASET_PROJECT,
            dataset_name=CLEARML_LANDMARK_DATASET_NAME
        )
        print(f"Found existing dataset '{dataset.name}' (ID: {dataset.id}). Will add/overwrite files.")
    except:
        print("Dataset not found, creating a new one.")
        dataset = Dataset.create(
            dataset_name=CLEARML_LANDMARK_DATASET_NAME,
            dataset_project=CLEARML_DATASET_PROJECT
        )

    # Save data to temporary files to add to ClearML Dataset
    temp_dir = "temp_upload_landmark_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define filenames as they will appear in the ClearML Dataset
    x_filename = "X_landmarks.npy"
    y_filename = "y_numeric_landmarks.npy"
    label_map_filename = "label_map_landmarks.npy"

    x_path = os.path.join(temp_dir, x_filename)
    y_path = os.path.join(temp_dir, y_filename)
    label_map_path = os.path.join(temp_dir, label_map_filename)

    np.save(x_path, X_data)
    np.save(y_path, y_numeric_data)
    np.save(label_map_path, label_to_id_map) # Save dict as 0-dim array

    print(f"Adding processed landmark files from '{temp_dir}' to the ClearML Dataset...")
    # Add files with specific names that will be used by downstream tasks
    dataset.add_files(path=temp_dir) # This adds all files in temp_dir

    print("Uploading dataset files to ClearML server...")
    dataset.upload(output_url=None)

    print("Finalizing dataset...")
    dataset.finalize()

    print(f"Landmark Feature Dataset '{CLEARML_LANDMARK_DATASET_NAME}' (ID: {dataset.id}) uploaded/updated and finalized successfully!")
    print(f"It contains: {x_filename}, {y_filename}, {label_map_filename}")

    # Clean up temporary files
    try:
        os.remove(x_path)
        os.remove(y_path)
        os.remove(label_map_path)
        os.rmdir(temp_dir)
        print("Temporary files cleaned up.")
    except OSError as e:
        print(f"Error removing temporary files: {e}")

if __name__ == "__main__":
    main()
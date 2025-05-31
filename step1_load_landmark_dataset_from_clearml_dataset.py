# step1_load_landmark_dataset_from_clearml_dataset.py
from clearml import Task, Dataset, Logger
import numpy as np
import os

# --- Configuration ---
CLEARML_LANDMARK_DATASET_PROJECT_SRC = "ASL_Classification_Pipeline"
CLEARML_LANDMARK_DATASET_NAME_SRC = "ASL_Landmark_Features_NPZ"

# This Task's details
TASK_PROJECT_NAME = "ASL_Classification_Pipeline"
TASK_NAME = "Pipeline Step 1: Load Landmark Dataset from ClearML_Dataset" # More specific name
# --- End Configuration ---

def main():
    # Initialize ClearML Task. This should be the first ClearML call in the script.
    task = Task.init(project_name=TASK_PROJECT_NAME, task_name=TASK_NAME)
    logger = task.get_logger() # Get the logger associated with this task
    
    task.execute_remotely()

    print(f"Attempting to get ClearML Landmark Feature Dataset: Project='{CLEARML_LANDMARK_DATASET_PROJECT_SRC}', Name='{CLEARML_LANDMARK_DATASET_NAME_SRC}'")
    try:
        landmark_feature_dataset = Dataset.get(
            dataset_project=CLEARML_LANDMARK_DATASET_PROJECT_SRC,
            dataset_name=CLEARML_LANDMARK_DATASET_NAME_SRC,
            # dataset_version="latest" # Or specify a particular version for reproducibility
        )
    except Exception as e:
        error_msg = f"Error getting landmark feature dataset '{CLEARML_LANDMARK_DATASET_NAME_SRC}': {e}. "\
                    f"Please ensure 'upload_dataset_for_landmarks.py' was run successfully and the dataset is finalized."
        logger.report_text(error_msg, level='error', print_console=True)
        raise ValueError(error_msg) # Or just raise, to propagate the original error

    print(f"Landmark Feature Dataset '{landmark_feature_dataset.name}' (ID: {landmark_feature_dataset.id}) retrieved. Getting local copy...")
    local_path_dataset_files = landmark_feature_dataset.get_local_copy()
    print(f"Dataset files available locally at: {local_path_dataset_files}")

    # Define filenames as they were saved in upload_dataset_for_landmarks.py
    x_filename_in_dataset = "X_landmarks.npy"
    y_filename_in_dataset = "y_numeric_landmarks.npy"
    label_map_filename_in_dataset = "label_map_landmarks.npy"

    try:
        X_raw = np.load(os.path.join(local_path_dataset_files, x_filename_in_dataset))
        y_numeric_raw = np.load(os.path.join(local_path_dataset_files, y_filename_in_dataset))
        label_map_loaded_raw = np.load(os.path.join(local_path_dataset_files, label_map_filename_in_dataset), allow_pickle=True)
        
        # Ensure label_map is a dictionary
        if label_map_loaded_raw.ndim == 0 and isinstance(label_map_loaded_raw.item(), dict):
            label_map = label_map_loaded_raw.item()
        elif isinstance(label_map_loaded_raw, dict): # Should ideally be saved as dict directly
            label_map = label_map_loaded_raw
        else: # Fallback if it's some other array-like structure that can be dict-ified
            try:
                label_map = dict(label_map_loaded_raw)
            except (TypeError, ValueError) as dict_conv_err:
                raise TypeError(f"Loaded label_map_loaded_raw (type: {type(label_map_loaded_raw)}) could not be converted to dict: {dict_conv_err}")

        print(f"Loaded data from ClearML Dataset: X shape: {X_raw.shape}, y_numeric shape: {y_numeric_raw.shape}")
        print(f"Loaded label map from ClearML Dataset: {label_map}")
    except Exception as e:
        error_msg = f"Error loading .npy files from dataset's local copy at '{local_path_dataset_files}': {e}"
        logger.report_text(error_msg, level='error', print_console=True)
        raise RuntimeError(error_msg)

    num_classes = len(label_map)
    logger.report_single_value(name="Dataset/Number of Classes", value=num_classes)
    logger.report_single_value(name="Dataset/Number of Features", value=X_raw.shape[1] if X_raw.ndim > 1 and X_raw.shape[1] > 0 else (1 if X_raw.ndim == 1 else 0))
    logger.report_single_value(name="Dataset/Total Samples Loaded", value=len(X_raw))

    # Upload these raw, un-split arrays as artifacts for the next step (splitting)
    task.upload_artifact(name='X_landmarks_raw', artifact_object=X_raw)
    task.upload_artifact(name='y_numeric_landmarks_raw', artifact_object=y_numeric_raw)
    task.upload_artifact(name='label_map_loaded', artifact_object=label_map) # Upload dict directly
    task.upload_artifact(name='num_classes_loaded', artifact_object=num_classes)

    # Log the ID of the ClearML Dataset version used for lineage and reproducibility
    task.connect_configuration(configuration={'source_clearml_dataset_id': landmark_feature_dataset.id,
                                            'source_clearml_dataset_version': landmark_feature_dataset.version},
                                name='Source ClearML Dataset Info')

    print("Pipeline Step 1 (Load Landmark Dataset from ClearML_Dataset) done. Raw landmark data uploaded as task artifacts.")

if __name__ == "__main__":
    main()

# step2_split_loaded_data.py
from clearml import Task, Logger
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
TASK_PROJECT_NAME = "ASL_Classification_Pipeline"
TASK_NAME = "Pipeline Step 2: Split Loaded Landmark Data"
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.1
RANDOM_STATE = 42
# --- End Configuration ---

def main():
    task = Task.init(project_name=TASK_PROJECT_NAME, task_name=TASK_NAME)
    logger = task.get_logger()
    task.execute_remotely()
    prev_step_task_id_manual = "c3e7b985b1ee4e61b0af07aee063e3b9" # <--- !!! FILL THIS MANUALLY FOR LOCAL TEST !!!

    # If run by a pipeline, it might pass the ID as a parameter.
    # Must match the parameter key defined in the pipeline step configuration.
    prev_step_task_id_from_pipeline = task.get_parameters_as_dict().get("Args/step1_load_task_id", None)

    # Prioritize ID from pipeline if available, otherwise use manual ID for local testing.
    prev_step_task_id = prev_step_task_id_from_pipeline if prev_step_task_id_from_pipeline else prev_step_task_id_manual
    
    if not prev_step_task_id or prev_step_task_id == "MANUALLY_RUN_PIPELINE_STEP1_TASK_ID_HERE":
        error_msg = ("Previous step (Load Landmark Dataset) task ID not provided or not updated for local testing. "
                    "Set 'prev_step_task_id_manual' in the script if running locally.")
        logger.report_text(error_msg, level='error', print_console=True)
        raise ValueError(error_msg)

    print(f"This task (Step 2 - Splitting) will use artifacts from Parent Task ID: {prev_step_task_id}")


    try:
        print(f"Fetching raw landmark data artifacts from parent task (ID: {prev_step_task_id})...")
        parent_task_obj = Task.get_task(task_id=prev_step_task_id)
        
        X_raw = parent_task_obj.artifacts['X_landmarks_raw'].get()
        y_numeric_raw = parent_task_obj.artifacts['y_numeric_landmarks_raw'].get()
        label_map = parent_task_obj.artifacts['label_map_loaded'].get()
        num_classes = int(parent_task_obj.artifacts['num_classes_loaded'].get())
        
        if not isinstance(label_map, dict):
            label_map = dict(label_map)

        print(f"Loaded raw data from parent task: X shape: {X_raw.shape}, y_numeric shape: {y_numeric_raw.shape}")
        print(f"Loaded label map from parent task: {label_map}, Num classes: {num_classes}")
    except Exception as e:
        error_msg = f"Error getting artifacts from parent task {prev_step_task_id}: {e}"
        logger.report_text(error_msg, level='error', print_console=True)
        raise RuntimeError(error_msg)

    # --- Data Splitting ---
    print("Splitting loaded landmark data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_numeric_raw, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y_numeric_raw
    )
    logger.report_single_value(name="Split Info/Test Set Size", value=len(X_test))
    print(f"Test set: X_test shape: {X_test.shape} (Size: {len(X_test)})")

    X_train, y_train = X_train_val, y_train_val
    X_val, y_val = None, None

    if VALIDATION_SPLIT_SIZE > 0 and len(X_train_val) > 1 :
        actual_validation_split_size = VALIDATION_SPLIT_SIZE
        if len(np.unique(y_train_val)) > 1 :
                X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=VALIDATION_SPLIT_SIZE, 
                random_state=RANDOM_STATE, 
                stratify=y_train_val
            )
        else:
            print("Warning: Not enough classes in y_train_val to stratify for validation split. Splitting without stratify.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=actual_validation_split_size, 
                random_state=RANDOM_STATE
            )
        
        logger.report_single_value(name="Split Info/Validation Set Size", value=len(X_val) if X_val is not None else 0)
        print(f"Validation set: X_val shape: {X_val.shape if X_val is not None else 'N/A'} (Size: {len(X_val) if X_val is not None else 0})")
        if X_val is not None and y_val is not None:
            task.upload_artifact('X_val_split', X_val)
            task.upload_artifact('y_val_split', y_val)
    else:
        logger.report_single_value(name="Split Info/Validation Set Size", value=0)
        print(f"No validation split performed based on VALIDATION_SPLIT_SIZE: {VALIDATION_SPLIT_SIZE}")
    
    logger.report_single_value(name="Split Info/Train Set Size", value=len(X_train))
    print(f"Train set: X_train shape: {X_train.shape} (Size: {len(X_train)})")

    task.upload_artifact('X_train_split', X_train)
    task.upload_artifact('y_train_split', y_train)
    task.upload_artifact('X_test_split', X_test)
    task.upload_artifact('y_test_split', y_test)
    task.upload_artifact('label_map_final', label_map) 
    task.upload_artifact('num_classes_final', num_classes)
    
    task.connect_configuration(configuration={
        'prev_step_load_task_id_used': prev_step_task_id,
        'test_split_fraction_config': TEST_SPLIT_SIZE,
        'validation_split_fraction_of_train_val_config': VALIDATION_SPLIT_SIZE if (VALIDATION_SPLIT_SIZE > 0 and len(X_train_val) > 1) else 0,
        'random_state_config': RANDOM_STATE
    }, name='Data Splitting Info')

    print("Pipeline Step 2 (Split Loaded Landmark Data) done. Split data uploaded as task artifacts.")

if __name__ == "__main__":
    main()

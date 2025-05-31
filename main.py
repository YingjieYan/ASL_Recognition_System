# main.py (Pipeline Controller for 3-Step Landmark-based MLP)
from clearml import Task
from clearml.automation import PipelineController

# --- Configuration ---
PIPELINE_PROJECT_NAME = "ASL_Classification_Pipeline" 
PIPELINE_NAME = "ASL_Landmark_MLP_3_Step_Processing_and_Training" 
PIPELINE_VERSION = "1.0.3" 

PIPELINE_RUNS_TARGET_PROJECT = None 
TARGET_EXECUTION_QUEUE = "pipeline" 
# --- End Configuration ---

def main():
    # 1.PipelineController
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PIPELINE_PROJECT_NAME,
        version=PIPELINE_VERSION,
    )

    pipe.set_default_execution_queue(TARGET_EXECUTION_QUEUE)
    print(f"Pipeline '{PIPELINE_NAME}' configured. Default execution queue: '{TARGET_EXECUTION_QUEUE}'.")
    print("IMPORTANT: Ensure the 'ASL_Landmark_Features_NPZ' ClearML Dataset (in project "
            f"'{PIPELINE_PROJECT_NAME}') exists. It should be created by running "
            "'upload_dataset_for_landmarks.py' manually first.")


    # Step 1: Load Landmark Dataset from ClearML Dataset
    step1_pipeline_step_name = "step1_load_landmark_data_from_cl_dataset"
    step1_base_task_name_in_script = "Pipeline Step 1: Load Landmark Dataset from ClearML_Dataset" 
    pipe.add_step(
        name=step1_pipeline_step_name, 
        base_task_project=PIPELINE_PROJECT_NAME, 
        base_task_name=step1_base_task_name_in_script 
    )
    print(f"Added Pipeline Step 1: '{step1_pipeline_step_name}' (based on Task '{step1_base_task_name_in_script}')")


    # Step 2: Split Loaded Data
    step2_pipeline_step_name = "step2_split_loaded_landmark_data"
    step2_base_task_name_in_script = "Pipeline Step 2: Split Loaded Landmark Data" 
    pipe.add_step(
        name=step2_pipeline_step_name,
        parents=[step1_pipeline_step_name], 
        base_task_project=PIPELINE_PROJECT_NAME,
        base_task_name=step2_base_task_name_in_script,
        parameter_override={
            "Args/step1_load_task_id": f"${{{step1_pipeline_step_name}.id}}"
        }
    )
    print(f"Added Pipeline Step 2: '{step2_pipeline_step_name}' (based on Task '{step2_base_task_name_in_script}')")


    # Step 3: Train & Evaluate Landmark MLP Model
    step3_pipeline_step_name = "step3_train_evaluate_landmark_mlp"
    step3_base_task_name_in_script = "Pipeline Step 3: Train & Evaluate Landmark MLP" 
    pipe.add_step(
        name=step3_pipeline_step_name,
        parents=[step2_pipeline_step_name],
        base_task_project=PIPELINE_PROJECT_NAME,
        base_task_name=step3_base_task_name_in_script,
        parameter_override={
            "Args/step2_split_task_id": f"${{{step2_pipeline_step_name}.id}}"
        }
    )
    print(f"Added Pipeline Step 3: '{step3_pipeline_step_name}' (based on Task '{step3_base_task_name_in_script}')")


    # --- 4. start Pipeline ---
    pipe.start(queue=TARGET_EXECUTION_QUEUE)
    print(f"Pipeline '{PIPELINE_NAME}' (version {PIPELINE_VERSION}) launched and enqueued on '{TARGET_EXECUTION_QUEUE}'.")
    print(f"Controller Task ID for this pipeline run: {pipe.id}")

if __name__ == "__main__":
    main()
from clearml import Task
from clearml.automation import PipelineController

PIPELINE_PROJECT_NAME = "ASL_Classification_Pipeline"
PIPELINE_NAME = "ASL_Landmark_MLP_Full_Pipeline_with_HPO"
PIPELINE_VERSION = "1.0.10"

PIPELINE_STEPS_EXECUTION_QUEUE = "pipeline"
HPO_TRIALS_EXECUTION_QUEUE = "pipeline"

def main():
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PIPELINE_PROJECT_NAME,
        version=PIPELINE_VERSION,
    )

    print(f"Pipeline '{PIPELINE_NAME}' (v{PIPELINE_VERSION}) configured.")

    pipe.add_step(
        name="Load_Landmark_Dataset",
        base_task_project=PIPELINE_PROJECT_NAME,
        base_task_name="Pipeline Step 1: Load Landmark Dataset from ClearML_Dataset",
        execution_queue=PIPELINE_STEPS_EXECUTION_QUEUE
    )

    pipe.add_step(
        name="Split_Landmark_Data",
        parents=["Load_Landmark_Dataset"],
        base_task_project=PIPELINE_PROJECT_NAME,
        base_task_name="Pipeline Step 2: Split Loaded Landmark Data",
        parameter_override={
            "Args/step1_load_task_id": "${Load_Landmark_Dataset.id}"
        },
        execution_queue=PIPELINE_STEPS_EXECUTION_QUEUE
    )

    pipe.add_step(
        name="Train_Base_MLP_Model",
        parents=["Split_Landmark_Data"],
        base_task_project=PIPELINE_PROJECT_NAME,
        base_task_name="Pipeline Step 3: Train & Evaluate Landmark MLP",
        parameter_override={
            "Args/step2_split_task_id": "${Split_Landmark_Data.id}"
        },
        execution_queue=PIPELINE_STEPS_EXECUTION_QUEUE
    )

    pipe.add_step(
        name="Optimize_MLP_Hyperparameters_Controller",
        parents=["Train_Base_MLP_Model"],
        base_task_project=PIPELINE_PROJECT_NAME,
        base_task_name="HPO: Train Model",
        parameter_override={
            "Args/base_train_task_id_for_hpo": "${Train_Base_MLP_Model.id}",
            "Args/project_name": PIPELINE_PROJECT_NAME,
            "Args/num_trials": 15,
            "Args/time_limit_minutes": 90,
            "Args/execution_queue_for_trials": HPO_TRIALS_EXECUTION_QUEUE,
            "Args/max_number_of_concurrent_tasks": 2,
            "Args/save_top_k_tasks_only": 3,
            "Args/objective_metric_title": "Validation_accuracy",
            "Args/objective_metric_series": "validation_accuracy",
            "Args/objective_metric_sign": "max"
        },
        execution_queue=PIPELINE_STEPS_EXECUTION_QUEUE
    )

    pipe.start(queue=PIPELINE_STEPS_EXECUTION_QUEUE)

    print(f"\nPipeline '{PIPELINE_NAME}' launched successfully.")
    print(f"Controller Task ID: {pipe.id}")
    print(f"View in ClearML UI under project: '{PIPELINE_PROJECT_NAME}'")

if __name__ == "__main__":
    main()

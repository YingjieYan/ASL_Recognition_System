# hpo_optimizer.py
from clearml import Task
from clearml.automation.optuna import OptimizerOptuna
# For older clearml versions, the import might be:

# --- Configuration ---
# 1. Project and Task Name for this HPO process
OPTIMIZER_TASK_PROJECT = "ASL_Classification_Pipeline" # Project for the HPO task itself
OPTIMIZER_TASK_NAME = "HPO for Landmark MLP Model"   # Name for this HPO task

# 2. Base Training Task ID (CRUCIAL)

BASE_TRAIN_TASK_ID = "ddcec9cd073a4a04a79254bb0591fd42" # <--- !!! IMPORTANT: Replace this !!!

# 3. Hyperparameters to Optimize

HYPER_PARAMETERS_TO_OPTIMIZE = [
    {
        "name": "Model Hyperparameters/learning_rate", # Full path to the hyperparameter in ClearML
        "type": "float",                               # Data type
        "min_val": 0.0001,                             # Minimum value for the search range
        "max_val": 0.01,                               # Maximum value for the search range
        "log_scale": True                              # Sample on a logarithmic scale
    },
    {
        "name": "Model Hyperparameters/hidden_layer_1_units", # Corresponds to 'hidden_layer_1_units'
        "type": "int",
        "min_val": 64,
        "max_val": 256,
        "step_size": 32                                # Step for integer sampling
    },
    {
        "name": "Model Hyperparameters/hidden_layer_2_units", # Corresponds to 'hidden_layer_2_units'
        "type": "int",
        "min_val": 32,
        "max_val": 128,
        "step_size": 32
    },
    {
        "name": "Model Hyperparameters/dropout_rate",       # Corresponds to 'dropout_rate'
        "type": "float",
        "min_val": 0.1,
        "max_val": 0.5,
        "step_size": 0.05 
    },
    {
        "name": "Model Hyperparameters/batch_size",         # Corresponds to 'batch_size'
        "type": "categorical",                             # Categorical parameter
        "values": [16, 32, 64]                             # List of discrete values to try
    }
]

# 4. Optimization Objective

OBJECTIVE_METRIC_TITLE = "Validation_accuracy" # MODIFIED: 'V' is capitalized, matching .capitalize()
OBJECTIVE_METRIC_SERIES = "validation_accuracy"  # This should be correct based on the callback
OBJECTIVE_METRIC_SIGN = "maximize"             # "maximize" or "minimize" the objective

# 5. Execution Configuration
NUM_CONCURRENT_WORKERS = 2  # Number of training tasks to run in parallel
MAX_ITERATIONS_PER_TASK = 1 # Each cloned task runs its full training once
TOTAL_MAX_JOBS = 20         # Total number of hyperparameter combinations to try
EXECUTION_QUEUE = "pipeline" # Queue where the cloned training tasks will be sent

# 6. Time Limit (Optional)
TIME_LIMIT_MINUTES = 120    # Total time limit for the HPO process in minutes (0 for no limit)
# --- End Configuration ---

def main():
    # Initialize the HPO task itself
    hpo_task = Task.init(project_name=OPTIMIZER_TASK_PROJECT, task_name=OPTIMIZER_TASK_NAME)
    hpo_logger = hpo_task.get_logger() # Use the HPO task's logger


    # Critical check for placeholder ID
    placeholder_base_id = "SUCCESSFUL_STEP3_TRAIN_TASK_ID_HERE"
    if BASE_TRAIN_TASK_ID == placeholder_base_id or not BASE_TRAIN_TASK_ID:
        error_message = (f"ERROR: Please replace 'BASE_TRAIN_TASK_ID' in the script "
                        f"(current value: '{BASE_TRAIN_TASK_ID}') with an actual ID of a "
                        f"successfully completed training task (e.g., "
                        f"'Pipeline Step 3: Train & Evaluate Landmark MLP'). "
                        f"This base training task will be cloned for hyperparameter optimization trials.")
        print(error_message)
        hpo_logger.report_text(error_message, level='error')
        return

    # Create the OptimizerOptuna object
    optimizer = OptimizerOptuna(
        base_task_id=BASE_TRAIN_TASK_ID,
        hyper_parameters=HYPER_PARAMETERS_TO_OPTIMIZE,
        objective_metric_title=OBJECTIVE_METRIC_TITLE,
        objective_metric_series=OBJECTIVE_METRIC_SERIES,
        objective_metric_sign=OBJECTIVE_METRIC_SIGN,
        

        max_number_of_concurrent_tasks=NUM_CONCURRENT_WORKERS,
        execution_queue=EXECUTION_QUEUE,
        max_iteration_per_job=MAX_ITERATIONS_PER_TASK,
        total_max_jobs=TOTAL_MAX_JOBS,
        
        check_job_every_seconds=15, # Check for completed jobs every 15 seconds (adjust as needed)
    )
    
    if TIME_LIMIT_MINUTES > 0:
        optimizer.set_time_limit(in_minutes=TIME_LIMIT_MINUTES)
        print(f"HPO time limit set to {TIME_LIMIT_MINUTES} minutes.")

    print("Starting Hyperparameter Optimization...")
    optimizer.start() # Starts the HPO process, enqueuing trial tasks
    print(f"Optimizer started. It will launch up to {TOTAL_MAX_JOBS} trial tasks on the '{EXECUTION_QUEUE}' queue.")
    print("Monitor the HPO progress and individual trials in the ClearML Web UI.")
    print(f"HPO Controller Task ID: {hpo_task.id}")


    print("HPO script has finished launching the optimizer. The HPO process runs on the ClearML server/agents.")

if __name__ == "__main__":
    main()
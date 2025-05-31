from clearml import Task
from clearml.automation import UniformParameterRange, DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optuna.optuna import OptimizerOptuna

# Initialize HPO controller task
task = Task.init(
    project_name="ASL_Classification_Pipeline",
    task_name="HPO: Train Model",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)
task.execute_remotely()

# Retrieve base task ID from pipeline input
args = task.get_parameters()
base_task_id = args.get("Args/base_train_task_id_for_hpo")
assert base_task_id, "Missing 'base_train_task_id_for_hpo' parameter."

# Configure and run hyperparameter optimization
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformParameterRange("Model Hyperparameters/learning_rate", min_value=0.0001, max_value=0.01),
        UniformParameterRange("Model Hyperparameters/dropout_rate", min_value=0.2, max_value=0.5),
        DiscreteParameterRange("Model Hyperparameters/batch_size", values=[16, 32, 64])
    ],
    objective_metric_title='Validation_accuracy',
    objective_metric_series='validation_accuracy',
    objective_metric_sign='max',
    optimizer_class=OptimizerOptuna,
    execution_queue='pipeline',
    max_iteration_per_job=10,
    total_max_jobs=20,
    max_number_of_concurrent_tasks=4,
    optimization_time_limit=60  # in seconds
)

optimizer.start()

from clearml import Task
from clearml.automation import UniformParameterRange, DiscreteParameterRange, HyperParameterOptimizer
from clearml.automation.optimization import Objective
from clearml.automation.optuna.optuna import OptimizerOptuna

# Create HPO task
task = Task.init(
    project_name="ASL_Classification_Pipeline",
    task_name="HPO: Train Model",
    task_type=Task.TaskTypes.optimizer
)

optimizer = HyperParameterOptimizer(
    base_task_id='7529baa1c68f43caba3ca1b6c5911618',  # training task ID (change id)
    hyper_parameters=[
        UniformParameterRange("General/learning_rate", min_value=0.0001, max_value=0.01),
        UniformParameterRange("General/dropout_rate", min_value=0.2, max_value=0.5),
        DiscreteParameterRange("General/batch_size", values=[16, 32, 64])
    ],
    objective_metric_title='Validation_accuracy',      
    objective_metric_series='validation_accuracy',     
    objective_metric_sign='max',                       
    optimizer_class=OptimizerOptuna,               # use Optuna as optimizer

    execution_queue='pipeline',
    max_iteration_per_job=10,
    total_max_jobs=20, 
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=60
)

# 启动优化过程
optimizer.start()

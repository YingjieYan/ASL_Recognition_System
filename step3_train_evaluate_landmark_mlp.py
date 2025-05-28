# step3_train_evaluate_landmark_mlp.py
from clearml import Task, Logger
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback as KerasCallback, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

# --- Configuration ---
TASK_PROJECT_NAME = "ASL_Classification_Pipeline"
TASK_NAME = "Pipeline Step 3: Train & Evaluate Landmark MLP"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_HIDDEN_LAYER_1_UNITS = 128
DEFAULT_HIDDEN_LAYER_2_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.3
OUTPUT_MODEL_FILENAME_ARTIFACT = "trained_asl_landmark_mlp_pipeline_final.keras"
OUTPUT_LABEL_MAP_ARTIFACT_NAME = "model_label_map_pipeline_final.npy"
# --- End Configuration ---

class ClearMLKerasProgressCallback(KerasCallback):
    def __init__(self, clearml_task_logger):
        super().__init__()
        self.logger = clearml_task_logger

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            series_name = metric.replace("val_", "validation_")
            self.logger.report_scalar(title=series_name.capitalize(), series=series_name, value=float(value), iteration=epoch)

def main():
    task = Task.init(project_name=TASK_PROJECT_NAME, task_name=TASK_NAME)
    logger = task.get_logger()

    hparams_dict_defaults = {
        'learning_rate': DEFAULT_LEARNING_RATE, 'batch_size': DEFAULT_BATCH_SIZE,
        'epochs': DEFAULT_EPOCHS, 'hidden_layer_1_units': DEFAULT_HIDDEN_LAYER_1_UNITS,
        'hidden_layer_2_units': DEFAULT_HIDDEN_LAYER_2_UNITS, 'dropout_rate': DEFAULT_DROPOUT_RATE,
    }
    effective_hparams = task.connect(hparams_dict_defaults)

    print(f"Effective Hyperparameters for this run: {effective_hparams}")

    current_lr = float(effective_hparams.get('learning_rate'))
    current_batch_size = int(effective_hparams.get('batch_size'))
    current_epochs = int(effective_hparams.get('epochs'))
    current_h1_units = int(effective_hparams.get('hidden_layer_1_units'))
    current_h2_units = int(effective_hparams.get('hidden_layer_2_units'))
    current_dropout = float(effective_hparams.get('dropout_rate'))

    # FOR LOCAL/MANUAL TESTING: Manually set the ID of a completed
    # "Pipeline Step 2: Split Loaded Landmark Data" task here.
    prev_step_task_id_manual = "0690b4b0a5b14697af7a0cfeb9447efb"  # <--- !!! FILL THIS MANUALLY FOR LOCAL TEST !!!
    
    # If run by a pipeline, it might pass the ID as a parameter.
    # The key "Args/step2_split_task_id" is an example.
    prev_step_task_id_from_pipeline = task.get_parameters_as_dict().get("Args/step2_split_task_id", None)
    
    prev_step_task_id = prev_step_task_id_from_pipeline if prev_step_task_id_from_pipeline else prev_step_task_id_manual

    if not prev_step_task_id or prev_step_task_id == "MANUALLY_RUN_PIPELINE_STEP2_TASK_ID_HERE":
        error_msg = ("Previous step (Split Landmark Data) task ID not provided or not updated for local testing. "
                    "Please set 'prev_step_task_id_manual' in the script if running locally.")
        logger.report_text(error_msg, level='error', print_console=True)
        raise ValueError(error_msg)

    print(f"This task (Step 3 - Training) will use artifacts from Parent Task ID: {prev_step_task_id}")


    try:
        print(f"Fetching split data artifacts from parent task (ID: {prev_step_task_id})...")
        parent_task_obj = Task.get_task(task_id=prev_step_task_id)
        
        X_train = parent_task_obj.artifacts['X_train_split'].get()
        y_train_numeric = parent_task_obj.artifacts['y_train_split'].get()
        X_test = parent_task_obj.artifacts['X_test_split'].get()
        y_test_numeric = parent_task_obj.artifacts['y_test_split'].get()
        
        label_map = parent_task_obj.artifacts['label_map_final'].get()
        num_classes = int(parent_task_obj.artifacts['num_classes_final'].get())

        X_val, y_val_numeric = None, None
        if 'X_val_split' in parent_task_obj.artifacts and 'y_val_split' in parent_task_obj.artifacts:
            X_val_artifact = parent_task_obj.artifacts['X_val_split']
            y_val_artifact = parent_task_obj.artifacts['y_val_split']
            if X_val_artifact and y_val_artifact: 
                X_val_candidate = X_val_artifact.get()
                y_val_numeric_candidate = y_val_artifact.get()
                if X_val_candidate is not None and y_val_numeric_candidate is not None and len(X_val_candidate) > 0:
                    X_val = X_val_candidate
                    y_val_numeric = y_val_numeric_candidate

        if not isinstance(label_map, dict):
            label_map = dict(label_map)

        print("Split data and label map loaded successfully.")

    except Exception as e:
        error_msg = f"Error getting artifacts from parent task {prev_step_task_id}: {e}"
        logger.report_text(error_msg, level='error', print_console=True)
        raise RuntimeError(error_msg)

    input_features = X_train.shape[1]
    y_train_categorical = to_categorical(y_train_numeric, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_numeric, num_classes=num_classes)
    validation_data_for_fit = None
    if X_val is not None and y_val_numeric is not None:
        y_val_categorical = to_categorical(y_val_numeric, num_classes=num_classes)
        validation_data_for_fit = (X_val, y_val_categorical)

    print("Defining Keras MLP model...")
    keras_model = Sequential([
        InputLayer(input_shape=(input_features,)),
        Dense(current_h1_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(current_dropout),
        Dense(current_h2_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(current_dropout),
        Dense(num_classes, activation='softmax')
    ])
    keras_model.compile(optimizer=Adam(learning_rate=current_lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    
    stringlist = []
    keras_model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_str = "\n".join(stringlist)
    logger.report_text(model_summary_str, title="Model Architecture Summary", print_console=False)
    print(model_summary_str)

    id_to_label_str_map = {str(v_id): label_str for label_str, v_id in label_map.items()}
    try:
        task.set_model_label_enumeration(id_to_label_str_map)
        print("Connected label enumeration to the Task for ClearML's auto-detected output models.")
    except AttributeError:
        logger.report_text("Warning: task.set_model_label_enumeration() method not found.", level='warning')
    except Exception as e_le:
        logger.report_text(f"Warning: Could not connect label enumeration to Task: {e_le}", level='warning')

    task.connect(keras_model, name="Keras_MLP_Auto_Log_Pipeline")

    print(f"Starting model training for {current_epochs} epochs with batch size {current_batch_size}...")
    callbacks_list = [ClearMLKerasProgressCallback(logger)]
    early_stopping_cb = EarlyStopping(
        monitor='val_loss' if validation_data_for_fit else 'loss',
        patience=15, restore_best_weights=True, verbose=1
    )
    callbacks_list.append(early_stopping_cb)

    history = keras_model.fit(
        X_train, y_train_categorical, batch_size=current_batch_size, epochs=current_epochs,
        validation_data=validation_data_for_fit, callbacks=callbacks_list, verbose=1
    )
    print("Model training complete.")

    print("Evaluating model on the test set...")
    loss, accuracy_keras = keras_model.evaluate(X_test, y_test_categorical, verbose=0)
    logger.report_scalar(title="Final Test Set Performance", series="Keras Loss", value=float(loss), iteration=1)
    logger.report_scalar(title="Final Test Set Performance", series="Keras Accuracy", value=float(accuracy_keras), iteration=1)
    print(f"Keras Model Evaluation on Test Set - Loss: {loss:.4f}, Accuracy: {accuracy_keras:.4f}")

    predictions_proba = keras_model.predict(X_test)
    predicted_classes_numerical = np.argmax(predictions_proba, axis=1)
    accuracy_sklearn = accuracy_score(y_test_numeric, predicted_classes_numerical)
    logger.report_scalar(title="Final Test Set Performance", series="Sklearn Accuracy", value=float(accuracy_sklearn), iteration=1)
    print(f"Sklearn Accuracy on test set: {accuracy_sklearn:.4f}")

    print("\nClassification Report (Test Set):")
    temp_id_to_label_map = {v: k for k, v in label_map.items()}
    class_names_for_report = [temp_id_to_label_map[i] for i in range(num_classes)]
    report_text = classification_report(y_test_numeric, predicted_classes_numerical, target_names=class_names_for_report, zero_division=0)
    print(report_text)
    logger.report_text(report_text, title="Classification Report (Test Set)", print_console=False)

    print("Generating and reporting confusion matrix to ClearML...")
    cm_display_labels = class_names_for_report
    cm = confusion_matrix(y_test_numeric, predicted_classes_numerical, labels=np.arange(num_classes))
    logger.report_confusion_matrix(
        "Confusion Matrix (Test Set)", "ignored_cm_series", matrix=cm,
        xaxis="Predicted Label", yaxis="Actual Label", iteration=1,
        xlabels=cm_display_labels, ylabels=cm_display_labels,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_display_labels)
    fig, ax = plt.subplots(figsize=(max(10, num_classes // 1.5), max(8, num_classes // 2)))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d')
    ax.set_title(f'Confusion Matrix (Test Set - Landmark MLP)')
    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="Confusion Matrix Plot (Test Set)", series="Confusion Matrix Figure",
        figure=fig, iteration=1, report_image=True
    )
    plt.close(fig)
    print("Confusion matrix reported to ClearML.")

    print(f"Saving trained model artifact: {OUTPUT_MODEL_FILENAME_ARTIFACT}")
    keras_model.save(OUTPUT_MODEL_FILENAME_ARTIFACT)
    task.upload_artifact(name="trained_landmark_model_file_artifact", artifact_object=OUTPUT_MODEL_FILENAME_ARTIFACT)

    print(f"Saving label map artifact: {OUTPUT_LABEL_MAP_ARTIFACT_NAME}")
    np.save(OUTPUT_LABEL_MAP_ARTIFACT_NAME, label_map)
    task.upload_artifact(name="model_label_map_artifact", artifact_object=OUTPUT_LABEL_MAP_ARTIFACT_NAME)

    final_summary = (
        f"Training & Evaluation Summary (Pipeline Step 3 - Landmark MLP):\n"
        f"Epochs Run (actual, due to EarlyStopping): {len(history.history.get('loss', [1]))}\n"
        f"Final Test Loss (Keras): {loss:.4f}\n"
        f"Final Test Accuracy (Keras): {accuracy_keras:.4f}\n"
        f"Final Test Accuracy (Sklearn): {accuracy_sklearn:.4f}\n"
        f"Manually uploaded model artifact: '{OUTPUT_MODEL_FILENAME_ARTIFACT}'\n"
        f"Manually uploaded label map artifact: '{OUTPUT_LABEL_MAP_ARTIFACT_NAME}'\n"
        f"ClearML auto-logged Keras model under 'Output Models' with connection name 'Keras_MLP_Auto_Log_Pipeline'."
    )
    logger.report_text(final_summary, title="Run Final Summary")
    print(final_summary)
    print("Pipeline Step 3 (Train & Evaluate Landmark MLP) completed successfully.")

if __name__ == "__main__":
    main()
from clearml import Task
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

task = Task.init(project_name="ASL_Classification", task_name="Pipeline Step 3: Train Model")

# task.execute_remotely()

# get pipeline parameters
args = Task.current_task().get_parameters_as_dict()
dataset_task_id = args.get("General/dataset_task_id", None)

if not dataset_task_id:
    dataset_task_id = "63ed35bdeed143c7913547413ed37b8b"  # 手动填写 fallback

task.connect({"General/dataset_task_id": dataset_task_id})

# get the last step of task
pre_task = Task.get_task(task_id=dataset_task_id)

# load artifacts: NumPy format training and testing sets
X_train = pre_task.artifacts['X_train'].get()
y_train = pre_task.artifacts['y_train'].get()
X_test = pre_task.artifacts['X_test'].get()
y_test = pre_task.artifacts['y_test'].get()


X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# build cnn model
model = Sequential([
    InputLayer(input_shape=(64, 64, 3)),
    Conv2D(32, (5, 5), activation='relu'), MaxPooling2D(), Dropout(0.25),
    Conv2D(64, (4, 4), activation='relu'), MaxPooling2D(), Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D(), Dropout(0.25),
    Conv2D(256, (2, 2), activation='relu'), MaxPooling2D(), Dropout(0.25),
    Flatten(), Dense(2048, activation='relu'), Dropout(0.25),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# model training
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# save and upload model
model_path = "asl_cnn_model.h5"
model.save(model_path)
task.upload_artifact('asl_model', artifact_object=model_path)

print(" Step 3 done. Model trained and uploaded.")

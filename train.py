import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the environment to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Image Data Generators for training and testing datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the training dataset
training_set = train_datagen.flow_from_directory(
    'dataSet/trainingData',
    target_size=(128, 128),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

# Prepare the testing dataset
test_set = test_datagen.flow_from_directory(
    'dataSet/testingData',
    target_size=(128, 128),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

# Build the model
classifier = Sequential()

# Add layers
classifier.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[128, 128, 1]))
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

classifier.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))

classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))

# Output layer with 27 units for classification (softmax for multi-class classification)
classifier.add(Dense(units=27, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
classifier.fit(training_set, epochs=5, validation_data=test_set)

# Save the model architecture and weights
model_json = classifier.to_json()
with open("Models1/model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')

classifier.save_weights('Models1/model_new.h5')
print('Weights saved')

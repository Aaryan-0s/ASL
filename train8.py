# Importing the Libraries
import tensorflow as tf

import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense , Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128


# Check TensorFlow version
print(tf.__version__)

# Part 1 - Data Preprocessing
classes_to_include = ["T", "K", "D","I"]
# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('data3/data2/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                  color_mode='grayscale',
                                                 classes=classes_to_include)

# Creating the Test set
test_set = test_datagen.flow_from_directory('data3/data2/test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical',
                                            color_mode='grayscale',
                                            classes=classes_to_include)

# Part 2 - Building the CNN

# Initializing the CNN
classifier = tf.keras.models.Sequential()

# Step 1 - Convolution
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))

# Step 2 - Pooling
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(tf.keras.layers.Flatten())

# Step 4 - Full connection
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 3 - Training the CNN

# Training the model
classifier.fit(training_set,
               epochs=5,
               steps_per_epoch=2802//32,
               validation_steps=930//32,
               validation_data=test_set)

# Part 4 - Saving the Model

# Saving the model to JSON and weights to H5
model_json = classifier.to_json()
with open("model-bw_tkdi.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw_tkdi.weights.h5')
print('Weights saved')

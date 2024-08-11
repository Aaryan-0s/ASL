# Importing the Libraries
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check TensorFlow version
print(tf.__version__)

# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('data2/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')

# Creating the Test set
test_set = test_datagen.flow_from_directory('data2/test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

# Part 2 - Building the CNN

# Initializing the CNN
classifier = tf.keras.models.Sequential()

# Step 1 - Convolution
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(tf.keras.layers.Flatten())

# Step 4 - Full connection
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=27, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 3 - Training the CNN

# Training the model
classifier.fit(training_set,
               epochs=5,
               validation_data=test_set)

# Part 4 - Saving the Model

# Saving the model to JSON and weights to H5
model_json = classifier.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model_new.weights.h5')
print('Weights saved')

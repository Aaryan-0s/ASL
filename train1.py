# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation="relu"))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation="relu"))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation="relu"))
classifier.add(Dense(units=3, activation="softmax"))  # softmax for more than 2

# Compiling the CNN
classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)  # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
classifier.summary()
# Code copied from - https://keras.io/preprocessing/image/
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

classes_to_include = ["D", "R", "U"]

training_set = train_datagen.flow_from_directory(
    "data2/train",
    target_size=(sz, sz),
    batch_size=10,
    color_mode="grayscale",
    class_mode="categorical",
    classes=classes_to_include,
)

test_set = test_datagen.flow_from_directory(
    "data2/test",
    target_size=(sz, sz),
    batch_size=10,
    color_mode="grayscale",
    class_mode="categorical",
    classes=classes_to_include,
)
classifier.fit(
    training_set,
    steps_per_epoch=1401 // training_set.batch_size,  # No of images in training set
    epochs=5,
    validation_data=test_set,
    validation_steps=465 // test_set.batch_size,  # No of images in test set
)  # No of images in test set


# Saving the model
model_json = classifier.to_json()
with open("model/" + "model-bw_dru.json", "w") as json_file:
    json_file.write(model_json)
print("Model Saved")
classifier.save_weights("model/" + "model-bw_dru.weights.h5")
print("Weights saved")

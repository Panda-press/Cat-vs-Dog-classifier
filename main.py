import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

train_features = []
train_targets = []

val_features = []
val_targets = []

test_features = []
test_targets = []

for i in range(0,10000):
    img = Image.open("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Dog\\"+str(i)+".jpg")
    train_features.append(np.array(img))
    train_targets.append([0])

for i in range(0,10000):
    img = Image.open("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Cat\\"+str(i)+".jpg")
    train_features.append(np.array(img))
    train_targets.append([1])    

train_features = np.array(train_features)
train_targets = np.array(train_targets)


for i in range(10000,11000):
    img = Image.open("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Dog\\"+str(i)+".jpg")
    val_features.append(np.array(img))
    val_targets.append([0])

for i in range(10000,11000):
    img = Image.open("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Cat\\"+str(i)+".jpg")
    val_features.append(np.array(img))
    val_targets.append([1])    

val_features = np.array(val_features)
val_targets = np.array(val_targets)


for i in range(11000,12000):
    img = Image.open("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Dog\\"+str(i)+".jpg")
    test_features.append(np.array(img))
    test_targets.append([0])

for i in range(11000,12000):
    img = Image.open("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Cat\\"+str(i)+".jpg")
    test_features.append(np.array(img))
    test_targets.append([1])    

test_features = np.array(test_features)
test_targets = np.array(test_targets)


print(train_features.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu, input_shape=(train_features[0].shape)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation = tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation = tf.nn.relu),  
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


model.fit(train_features, train_targets, batch_size = 200, epochs = 20, shuffle = True, validation_data = (val_features, val_targets))

model.evaluate(test_features, test_targets)
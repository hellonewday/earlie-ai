# import tensorflow
# from tensorflow import keras
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

datasetType = os.listdir("../../data/food-5k")
classes = os.listdir("../../data/food-5k/training")
print(datasetType)
# one_hot = to_categorical(classes)
# print(one_hot)
train_dataGen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=20, featurewise_center=True,
                                   featurewise_std_normalization=True)

val_dataGen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 rotation_range=20, featurewise_center=True,
                                 featurewise_std_normalization=True)

img_size = (224, 224)
batch_size = 30
epochs = 10
train_data = train_dataGen.flow_from_directory("../../data/food-5k/training", target_size=img_size,
                                               batch_size=batch_size,
                                               class_mode="binary")

# test_data = dataGen.flow_from_directory("../../data/food-5k/evaluation", target_size=img_size, batch_size=batch_size,
#                                         class_mode="categorical", subset="evaluation")

val_data = val_dataGen.flow_from_directory("../../data/food-5k/validation", target_size=img_size, batch_size=batch_size,
                                           class_mode="binary")

classesName = list(train_data.class_indices.keys())
print("Classes: ", classesName)

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=train_data.image_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=2e-5), metrics="accuracy")

print(model.summary())

history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size,
    epochs=epochs)

epoch_range = range(1, epochs + 1)

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epochs, training_loss)
plt.plot(epochs, validation_loss)
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

model.save("food-5k.h5")

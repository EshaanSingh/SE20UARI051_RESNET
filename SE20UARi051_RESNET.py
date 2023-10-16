!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json

api_token = {"username":"vedantdhanyamraju","key":"fa61b1588715a06f72326a191014b836"}

import json

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d tongpython/cat-and-dog --force
!unzip cat-and-dog.zip
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.metrics import Accuracy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
train_dir = os.path.join('training_set', 'training_set')
test_dir = os.path.join('test_set', 'test_set')
trainloader = image_dataset_from_directory(train_dir, labels='inferred', batch_size=32, image_size=(224,224), shuffle=True, seed=42)
testloader = image_dataset_from_directory(test_dir, labels='inferred', batch_size=32, image_size=(224,224), shuffle=True, seed=42)
val_batches = tf.data.experimental.cardinality(testloader)
valloader = testloader.take(val_batches//5)
testloader = testloader.skip(val_batches//5)
class_names = trainloader.class_names
print(class_names)
plt.figure(figsize=(10, 10))
for images, labels in trainloader.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
resnet = ResNet50(input_shape=(224,224,3),
                  include_top = False,
                  weights='imagenet')
resnet.trainable=False
image_batch, label_batch = next(iter(trainloader))
feature_batch = resnet(image_batch)
print(feature_batch.shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(224,224,3)))
model.add(resnet)
model.add(global_average_layer)
model.add(tf.keras.layers.Dropout(0.2))
model.add(prediction_layer)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])
initial_epochs = 15
loss0, accuracy0 = model.evaluate(valloader)
print("initial loss: ", loss0)
print("initial accuracy: ", accuracy)
hist = model.fit(trainloader,
                 epochs = initial_epochs,
                 validation_data = valloader)
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
accuracy = Accuracy()

for x,y in testloader:
  prediction = model.predict(x).flatten()
  prediction = tf.nn.sigmoid(prediction)
  prediction = tf.where(prediction < 0.5, 0, 1)
  accuracy.update_state(y, prediction)
accuracy = accuracy.result()
print("Testing Accuracy: " ,accuracy.numpy())
image_batch, label_batch = testloader.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

label_batch = ["cats" if i==0 else "dogs" for i in label_batch]
pred_labels = ["cats" if i==0 else "dogs" for i in predictions]

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
    

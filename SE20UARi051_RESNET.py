import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained ResNet model
model = ResNet50(weights='imagenet', include_top=False)

# Freeze the first few layers of the model
for layer in model.layers[:5]:
    layer.trainable = False

# Add a new classification layer to the model
x = model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)

# Compile the model
model = tf.keras.Model(inputs=model.input, outputs=x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare the data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'train_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'test_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# Train the model
model.fit(train_generator,
          epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test accuracy:', test_accuracy)

# Use the model to classify new images
new_image = tf.keras.preprocessing.image.load_img('new_image.jpg', target_size=(224, 224))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = tf.expand_dims(new_image, axis=0)

predictions = model.predict(new_image)

if predictions[0][0] > predictions[0][1]:
    print('The image is a cat.')
else:
    print('The image is a dog.')

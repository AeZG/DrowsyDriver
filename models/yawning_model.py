import tensorflow as tf
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ----------------------------
# 1. Define Data Directories
# ----------------------------
data_dir = 'C:/Users/Greg/PycharmProject/DrowsyDriver/datasets/mouth_dataset'

# ----------------------------
# 2. Define Parameters
# ----------------------------
IMG_SIZE = (64, 64)
batch_size = 32

# ----------------------------
# 3. Create Data Generators (Grayscale) with Data Augmentation
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),  # Adjust brightness
    validation_split=0.2  # 20% for validation
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
    subset='training'
)

test_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
    subset='validation'
)

# ----------------------------
# 4. Build the CNN Model
# ----------------------------
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 5. Train the Model
# ----------------------------
epochs = 10
steps_per_epoch = len(train_generator)
validation_steps = len(test_generator)
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps
)

# Save the model
model.save('yawn_detection_model.h5')

# ----------------------------
# 6. Evaluate and Visualize Training Results
# ----------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

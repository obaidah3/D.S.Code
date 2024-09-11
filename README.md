# D.S.Code
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Explore the dataset
print(f"Training data shape: {train_images.shape}")
print(f"Testing data shape: {test_images.shape}")
print(f"Unique labels: {np.unique(train_labels)}")

# Visualize some examples from the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Reshape input data for CNN
train_images_cnn = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images_cnn = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Define the CNN model architecture
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the CNN model
cnn_model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# Flatten input data for ANN
train_images_ann = train_images.reshape((train_images.shape[0], 28 * 28))
test_images_ann = test_images.reshape((test_images.shape[0], 28 * 28))

# Define the ANN model architecture
ann_model = Sequential([
    Dense(512, activation='relu', input_shape=(28 * 28,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the ANN model
ann_model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the CNN model
cnn_history = cnn_model.fit(train_images_cnn, train_labels, epochs=10, 
                            validation_data=(test_images_cnn, test_labels))

# Train the ANN model
ann_history = ann_model.fit(train_images_ann, train_labels, epochs=10, 
                            validation_data=(test_images_ann, test_labels))

                            
# Evaluate the CNN model
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_images_cnn, test_labels, verbose=2)
print(f"Test accuracy for CNN: {cnn_test_acc:.4f}")

# Evaluate the ANN model
ann_test_loss, ann_test_acc = ann_model.evaluate(test_images_ann, test_labels, verbose=2)
print(f"Test accuracy for ANN: {ann_test_acc:.4f}")

# Plot training & validation accuracy for comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Accuracy')
plt.plot(ann_history.history['accuracy'], label='ANN Train Accuracy')
plt.plot(ann_history.history['val_accuracy'], label='ANN Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot training & validation loss for comparison
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='CNN Train Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Val Loss')
plt.plot(ann_history.history['loss'], label='ANN Train Loss')
plt.plot(ann_history.history['val_loss'], label='ANN Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()                  
                  

# Knee Osteoporosis X-ray Classification - README

## About the Project
This project involves developing a deep learning model to classify knee X-ray images for the detection of osteoporosis. The dataset used is the "Multi-class Knee Osteoporosis X-ray Dataset" available on Kaggle. The goal is to build a convolutional neural network (CNN) capable of predicting the category of an X-ray image.

## Technologies Used
- Python 3
- TensorFlow & Keras
- NumPy
- Matplotlib

## Dataset Preparation
The dataset consists of categorized knee X-ray images labeled for different levels of osteoporosis. It is divided into training and validation sets using `ImageDataGenerator` with a 90%-10% split.

```python
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
```

The images are resized to 150x150 pixels and normalized by rescaling the pixel values between 0 and 1.

## Model Architecture
The model is a Convolutional Neural Network (CNN) designed using Keras' Sequential API. It includes three convolutional layers with ReLU activation and max pooling, followed by a dense layer with 512 units, and a softmax output layer for classification.

```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])
```

This architecture is suitable for image classification tasks and can be further optimized with techniques such as dropout or batch normalization.

## Training the Model
The model is trained over 10 epochs using the Adam optimizer and categorical crossentropy loss, which is appropriate for multi-class classification problems:

```python
model.fit(train_generator, validation_data=validation_generator, epochs=10)
```

## Saving the Model
After training, the model is saved in HDF5 format for later use:

```python
model.save("image_classifier.h5")
```

## Making Predictions
A utility function is provided to predict the class of a single image and visualize the result. It loads an image, preprocesses it, feeds it into the model, and then displays the predicted label.

```python
def predict_image(image_path, model, class_indices):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class]

    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis("off")
    plt.show()
```

## Results and Future Work
The CNN demonstrated meaningful performance on the validation set, indicating its potential usefulness for medical image classification tasks. Future enhancements could include:
- **Data Augmentation**: To improve generalization and reduce overfitting.
- **Transfer Learning**: Utilizing pre-trained models (e.g., VGG16, ResNet) to leverage learned features.
- **Hyperparameter Tuning**: Experimenting with different learning rates, batch sizes, and layer configurations.
- **Model Evaluation**: Integrating confusion matrices, classification reports, and ROC curves for deeper insights.

This project showcases the effectiveness of deep learning techniques in automating and improving medical diagnostics through image analysis.


import numpy as np
import cv2
from tensorflow.keras.models import load_model

def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)   
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict_stroke(image):
    model = load_model('image_models/Brain_Stroke_CNN/brain_stroke_model.keras')
    prediction = model.predict(image)
    predicted_label = (prediction > 0.5).astype(int)[0][0]  
    return predicted_label

def CNN_Model(image_path):
    image = preprocess_image(image_path)
    predicted_label = predict_stroke(image)

    if predicted_label == 0:
        print(f'Predicted label: {predicted_label} -> Predicted class: hemorrhagic')
        return 0
    else:
        print(f'Predicted label: {predicted_label} -> Predicted class: ischaemic')
        return 1
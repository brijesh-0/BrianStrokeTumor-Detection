import numpy as np
import cv2
from tensorflow.keras.models import load_model

image_path = '/content/Brain_Stroke_CT-SCAN_image 2/Test/ischaemic/99 (9).jpg'

def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)   
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict_stroke(image):
    model = load_model('brain_stroke_model.keras')
    prediction = model.predict(image)
    predicted_label = (prediction > 0.5).astype(int)[0][0]  
    return predicted_label

image = preprocess_image(image_path)
predicted_label = predict_stroke(image)

if predicted_label == 0:
        print(f'Predicted label: {predicted_label} -> Predicted class: hemorrhagic')
else:
    print(f'Predicted label: {predicted_label} -> Predicted class: ischaemic')
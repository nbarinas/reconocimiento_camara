import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar el modelo de predicción de edad
model_path = 'path_to_your_age_model.h5'
age_model = load_model(model_path)

def predict_age(face_image):
    # Preprocesar la imagen para el modelo
    face_image = cv2.resize(face_image, (200, 200))
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image / 255.0  # Normalizar
    
    # Predecir la edad
    age_prediction = age_model.predict(face_image)
    return int(age_prediction[0][0])

def video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        status, frame = cap.read()
        
        if not status:
            break
        
        # Detectar las caras en el frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            face_image = rgb_frame[top:bottom, left:right]
            
            # Estimar la edad
            try:
                age = predict_age(face_image)
                age_text = f"Age: {age}"
                
                # Dibujar un rectángulo alrededor de la cara y mostrar la edad
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, age_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print("Error en la estimación de edad:", e)

        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()

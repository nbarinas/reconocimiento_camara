import cv2
import mediapipe as mp

# Inicializar MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils  # Utilidades para dibujar puntos clave
mp_pose = mp.solutions.pose  # Solución de pose de MediaPipe

def video():
    cap = cv2.VideoCapture(0)  # Capturar video desde la cámara por defecto (índice 0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            status, frame = cap.read()
            
            if not status:
                break
            
            # Convertir la imagen de BGR a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Mejorar el rendimiento marcando la imagen como no editable
            
            # Hacer la detección
            results = pose.process(image)
            
            # Convertir la imagen de nuevo a BGR para OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Dibujar las anotaciones de los puntos clave en la imagen
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),  # Estilo de los puntos
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)   # Estilo de las conexiones
                )
            
            # Mostrar el cuadro con los puntos clave del cuerpo
            cv2.imshow('Pose Estimation', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video()

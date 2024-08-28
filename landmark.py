import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def video():
    cap = cv2.VideoCapture(0)
    
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():
        status, frame = cap.read()
        
        if not status:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        # Configurar estilo de dibujo para puntos y conexiones
        face_landmarks_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        face_connections_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        
        if results.face_landmarks:
            # Solo dibuja algunos landmarks
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                if idx % 5 == 0:  # Por ejemplo, dibuja solo cada 5º punto
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Dibuja las conexiones de la malla facial
            mp_drawing.draw_landmarks(
                frame, 
                results.face_landmarks, 
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connections_style
            )
        
        # Dibuja los landmarks de la pose y manos como antes
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()

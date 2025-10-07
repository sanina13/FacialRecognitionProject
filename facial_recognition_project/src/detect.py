import cv2
import mediapipe as mp

# Start modules detection Faces MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#Video from webcam (0 = webcam)
cap = cv2.VideoCapture(1)

with mp_face_detection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break


        # Convert from OpenCV to MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Process de Image to detect faces
        results = face_detection.process(rgb_frame)

        # If exists faces MediaPipe draw them

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)


        # Shows the result

        cv2.imshow('Facial Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27: # tecla ESC para sair
            break


cap.release()
cv2.destroyAllWindows()

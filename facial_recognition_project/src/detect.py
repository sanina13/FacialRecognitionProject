import cv2
import mediapipe as mp

# Start modules detection Faces MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#Video from webcam (0 = webcam)

def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise ValueError("Erro ao abrir webcam.")

    with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
        
        print('Webcam iniciada. Pressionar ESC para sair.')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                raise ValueError('Erro ao ler o frame da câmera')
            

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

if __name__ == "__main__":
    main()

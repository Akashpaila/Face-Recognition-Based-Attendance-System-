import cv2
import numpy as np
import dlib

LEFT_EYE_START, LEFT_EYE_END = 42, 47
RIGHT_EYE_START, RIGHT_EYE_END = 36, 41
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Function to calculate eye aspect ratio (EAR)xx
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
blink_status = False
consecutive_frames = 0

cap = cv2.VideoCapture(0)

text = "FAKE"

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(LEFT_EYE_START, LEFT_EYE_END + 1)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(RIGHT_EYE_START, RIGHT_EYE_END + 1)])

        # Calculate eye aspect ratio (EAR) for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if the average eye aspect ratio is below the threshold
        if avg_ear < EYE_AR_THRESH:
            consecutive_frames += 1
            if consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                blink_status = True
                text = "REAL"
        else:
            consecutive_frames = 0
            blink_status = False

        # Draw rectangles around the eyes
        left_eye_rect = cv2.boundingRect(left_eye)
        right_eye_rect = cv2.boundingRect(right_eye)
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (255, 0, 0), 2)

    # Display the text
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Eye Blink Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

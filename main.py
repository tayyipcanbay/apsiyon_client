import numpy as np
import cv2
import time
import requests
import threading
from cv2 import imencode

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Initialize the QRCode detector
qr_detector = cv2.QRCodeDetector()

# Variable to store execution state
qr_detected = False
qr_detection_start_time = None

face_detection_start = False

req_in_progress = False
response = None

authorized = False
authorized_display_start_time = None
decoded_info = None
qr_content = None

# Function to send a POST request with the detected face image and decoded QR content
def auth_request(image, token):
    global response, req_in_progress, authorized
    try:
        # Encode the image to a format suitable for transmission
        _, image_encoded = imencode('.jpg', image)
        image_bytes = image_encoded.tobytes()

        headers = {'Authorization': token}
        files = {'image': ('face.jpg', image_bytes, 'image/jpeg')}
        
        response = requests.post("https://httpbin.org/delay/3", headers=headers, files=files, verify=False)
        
        if response.status_code == 200:
            response_text = "Request successful!"
            authorized = True
        else:
            response_text = "Request failed!"
        
        response = response_text
    except requests.RequestException as e:
        response = f"Request failed: {e}"
    req_in_progress = False

# Starting the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Converting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Check for QR code
    decoded_info, points, _ = qr_detector.detectAndDecode(gray)
    if not face_detection_start:
        if decoded_info:
            print(f'Decoded Info : \n {decoded_info}')
            if not qr_detected:
                # Start the timer when QR code is detected for the first time
                qr_detection_start_time = time.time()
                qr_detected = True
                qr_content = decoded_info
                print(qr_content)
            else:
                # QR detection has lasted for 3 seconds
                cv2.putText(img, "Starting face detection...", (170, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                # Proceed to face detection or other processing here
                face_detection_start = True
        else:
            qr_detected = False
            qr_detection_start_time = None
            cv2.putText(img, "QR Code not detected", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    else:
        if authorized:
            if authorized_display_start_time is None:
                authorized_display_start_time = time.time()

            elapsed_time = time.time() - authorized_display_start_time
            if elapsed_time < 3:
                cv2.putText(img, "AUTHORIZED", (70, 160), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            else:
                # Reset states for the next QR code detection
                authorized = False
                authorized_display_start_time = None
                face_detection_start = False
                qr_detected = False
                qr_detection_start_time = None
        else:
            # Detecting the face for region of image to be fed to eye classifier
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
            eyes = []  # Initialize eyes variable
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # roi_face is face which is input to eye classifier
                    roi_face = gray[y:y + h, x:x + w]
                    roi_face_clr = img[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

                    # Examining the length of eyes object for eyes
                    if len(eyes) >= 2:
                        cv2.putText(img, "Eyes open!", (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    else:
                        # Blink detected, initiate ping
                        print("Blink detected--------------")
                        if not req_in_progress and not authorized:
                            req_in_progress = True
                            response = None
                            threading.Thread(target=auth_request, args=(roi_face_clr, decoded_info)).start()

                        # Display waiting message while ping is in progress
                        if req_in_progress:
                            cv2.putText(img, "Pinging server...", (70, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                        else:
                            # Display the result of the ping
                            if response:
                                print(response)
                                cv2.putText(img, response, (70, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0) if "successful" in response else (0, 0, 255), 2)
                                if authorized:
                                    cv2.putText(img, "AUTHORIZED", (70, 160), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            else:
                cv2.putText(img, "No face detected", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            # Add animations for blink detection
            if len(faces) > 0 and len(eyes) < 2:
                cv2.putText(img, "Blink Detected", (70, 190), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                for i in range(5):
                    img = cv2.putText(img, ".", (200 + i*20, 190), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('QR Code and Face Detection', img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

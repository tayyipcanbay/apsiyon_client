import cv2
import time
import requests
import threading

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Initialize the QRCode detector
qr_detector = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)


# Mode Flags
qr_read_mode = True
face_detection_mode = False
auth_mode = False
authorized = False


# Data
response = None
decoded_info = None
blink_count = 0
img_face = None
img_wide = None

# Thread Flags
req_in_progress = False

def auth_request(token, img_face, img_wide):
    global req_in_progress, response, authorized

    url = 'https://httpbin.org/delay/3'
    headers = {
        'Authorization': f'{token}'
    }

    if img_face is not None:
        _, img_face_encoded = cv2.imencode('.jpg', img_face)
        img_face_bytes = img_face_encoded.tobytes()
    else:
        img_face_bytes = None

    if img_wide is not None:
        _, img_wide_encoded = cv2.imencode('.jpg', img_wide)
        img_wide_bytes = img_wide_encoded.tobytes()
    else:
        img_wide_bytes = None

    files = {
        'img_face': ('img_face.jpg', img_face_bytes, 'image/jpeg') if img_face_bytes is not None else None,
        'img_wide': ('img_wide.jpg', img_wide_bytes, 'image/jpeg') if img_wide_bytes is not None else None
    }

    files = {k: v for k, v in files.items() if v is not None}

    response = requests.post(url, headers=headers, files=files)
    req_in_progress = False
    authorized = False

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    if qr_read_mode :
        decoded_info, points, _ = qr_detector.detectAndDecode(gray)

        if decoded_info :
            qr_read_mode = False
            face_detection_mode = True

    if face_detection_mode : 
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
            eyes = []
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    roi_face = gray[y:y + h, x:x + w]
                    roi_face_clr = img[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

                    if len(eyes) >= 2:
                        cv2.putText(img, 'EYES OPEN' + str(blink_count), (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    else:
                        blink_count += 1
                        cv2.putText(img, 'BLINK COUNT' + str(blink_count), (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    
                    if blink_count >= 10 and  len(eyes) >= 2: 
                        img_face = img[y:y + h, x:x + w]
                        img_wide = img

                        auth_mode = True
                        face_detection_mode = False
    
    if auth_mode : 
        if not req_in_progress and not authorized and response is None: 
            req_in_progress = True
            response = None
            threading.Thread(target=auth_request, args=(decoded_info, img_face, img_wide)).start()
            print("Request Sent, waiting for response")

        if req_in_progress : 
            cv2.putText(img, 'REQUEST...', (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        else:
            if authorized : 
                cv2.putText(img, 'Authorized', (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            else : 
                cv2.putText(img, 'Access Denied', (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            

    cv2.imshow('QR Code and Face Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

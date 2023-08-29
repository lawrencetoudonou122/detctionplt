import cv2
import numpy as np
import easyocr
import urllib.parse
import os
camera_ip = "192.168.1.64"
username = "admin"
password = "Scita@123"
encoded_username = urllib.parse.quote(username)
encoded_password = urllib.parse.quote(password)
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

reader = easyocr.Reader(['en'])

rtsp_url = f"rtsp://{encoded_username}:{encoded_password}@{camera_ip}"
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    detections_layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(detections_layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = obj[0:4] * np.array([W, H, W, H])
                x_center, y_center, width, height = box.astype('int')
                x_min = int(x_center - (width / 2))
                y_min = int(y_center - (height / 2))

                roi = frame[y_min:y_min + int(height), x_min:x_min + int(width)]

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(roi_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                output = reader.readtext(thresh)

                for (bbox, text, prob) in output:
                    if prob > 0.4:
                        text = f"{text} ({prob:.2f})"
                        cv2.rectangle(frame, (x_min, y_min), (x_min + int(width), y_min + int(height)), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

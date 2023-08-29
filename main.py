import os
import cv2
import numpy as np
import easyocr
import util

# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

camera_ip = "192.168.1.64"
username = "admin"
password = "Scita@123"
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
cap = cv2.VideoCapture(f"rtsp://{username}:{password}@{camera_ip}/h264_ulaw.sdp")
reader = easyocr.Reader(['en'])
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True)
    net.setInput(blob)
    detections = util.get_outputs(net)
    bboxes = []
    class_ids = []
    scores = []
    for detection in detections:
        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        license_plate = frame[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
        frame = cv2.rectangle(frame,
                              (int(xc - (w / 2)), int(yc - (h / 2))),
                              (int(xc + (w / 2)), int(yc + (h / 2))),
                              (0, 255, 0),
                              15)
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
        output = reader.readtext(license_plate_thresh)
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                print(text, text_score)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

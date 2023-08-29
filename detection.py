-import cv2
import numpy as np
import easyocr
import urllib.parse 
import util
import os

username = "admin"
password = "Scita@123"
encoded_password = urllib.parse.quote(password, safe='')

camera_ip = "192.168.1.64"
gstreamer_str = (
    f"rtspsrc location=rtsp://{username}:{encoded_password}@{camera_ip}/h264_ulaw.sdp latency=100 ! queue ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGR ! "
    "appsink drop=1"
)
cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
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

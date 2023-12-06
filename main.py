import cv2
import numpy as np
from picamera2 import Picamera2
from gtts import gTTS
import os

Width=448
Height=448
wh = 320
conf_threshold = 0.5
nms_threshold = 0.3

classes = []
with open('yolov3.txt','rt') as f:
    classes = [line.strip() for line in f.readlines()]
    
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
print(classes)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (Width, Height)}))
picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
picam2.start()

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layerNames = net.getLayerNames()
outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(label)
    
    myobj = gTTS(text=label, lang='en', slow=False)
    myobj.save("audio.mp3")
    os.system("mpg123 audio.mp3")

        
while True:
    image = picam2.capture_array()
 
    blob = cv2.dnn.blobFromImage(image, 1/255, (wh,wh), [0,0,0], crop=False)
    net.setInput(blob)
    outs = net.forward(outputNames)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    if 1:
        cv2.imshow('Image', image)
        cv2.waitKey(10)
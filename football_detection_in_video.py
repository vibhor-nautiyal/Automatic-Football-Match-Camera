import cv2
import numpy as np
import glob
import random
import time
# from PCA9685 import PCA9685
# pwm = PCA9685(0x40, debug=False)
# pwm.setPWMFreq(50)
# pwm.setServoPosition(0, 90)
# Load Yolo
net = cv2.dnn.readNet("D:\\test\practice\project\\train_yolo_to_detect_custom_object\yolo_custom_detection\yolov3_training_1000.weights", "D:\\test\practice\project\\train_yolo_to_detect_custom_object\yolo_custom_detection\yolov3_testing.cfg")

# Name custom object
classes = ["football"]

# Images path
# images_path = glob.glob(r"D:\\test\\practice\\project\dataset\\images\*.png")
# print(images_path)

cap=cv2.VideoCapture("D:\\test\\practice\\project\dataset\\test2.mp4")


#servo parameters
_, frame = cap.read()
rows, cols, _ = frame.shape
x_medium = int(cols / 2)
center = int(cols / 2)
position = 90 # degrees


layer_names = net.getLayerNames()
# print(layer_names)
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
# random.shuffle(images_path)
# loop through all the images
# for img_path in images_path:
#     # Loading image
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, None, fx=0.4, fy=0.4)
#     height, width, channels = img.shape
while(True):
    
    ret,img=cap.read()
    if(not ret):
        break
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    # font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # color = colors[class_ids[i]]
            # print(color)
            color=[0,0,0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            x_medium = int((x + x + w) / 2)
            cv2.line(img, (x_medium, 0), (x_medium, height), (0, 0, 0), 2)
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
            if x_medium < center -30:
                position += 1
            elif x_medium > center + 30:
                position -= 1
            # pwm.setServoPosition(0, position)

    cv2.imshow("Image", img)
    # time.sleep(0.001)
    key = cv2.waitKey(1)
    if key==27:
        break

cv2.destroyAllWindows()
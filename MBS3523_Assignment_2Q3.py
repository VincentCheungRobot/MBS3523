import cv2
import numpy as np
confThreshold = 0.8
cam = cv2.VideoCapture(0)
cam.set(3, 960)
cam.set(4, 540)
classesFile = 'coco80.names'
classes = None
with open(classesFile, 'r') as f:
    classes = f.read().rstrip('\n').split('\n')
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
fruit_prices = {'apple': 1, 'banana': 2, 'orange': 3}
fruit_colors = {"apple": (0, 255, 0), "banana": (0, 255, 255),"orange": (0, 165, 255)}


while True:
    success, img = cam.read()
    height, width, ch = img.shape
    fruit_counter = {fruit: 0 for fruit in fruit_prices.keys()}
    total_price = 0

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    LayerOutputs = net.forward(net.getUnconnectedOutLayersNames())
    class_ids = []
    confidences = []
    bboxes = []

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:]  # omit the first 5 values
            class_id = np.argmax(scores)  # find the highest score ID out of 80 values which has the highest confidence value
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0] * width)  # YOLO predicts centers of image
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                font = cv2.FONT_HERSHEY_PLAIN



    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.5)
    # print(len(indexes))
    # print(indexes)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = fruit_colors[label] if label in fruit_colors else (0, 0, 255)  # Red color for unknown fruits
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
            if label in fruit_prices:
                fruit_counter[label] += 1
                total_price += fruit_prices[label]
                # print(fruit_prices[label])

    for i, (fruit, count) in enumerate(fruit_counter.items()):
        cv2.putText(img, f'{fruit}: {count}', (width - 200, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.67,
            (0, 0, 255), 1)
        cv2.putText(img, f'Total price: {total_price}', (width - 250, 30 + len(fruit_counter) * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 255), 1)

    cv2.putText(img, "Price List: apple$1 banana$2 orange$3", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
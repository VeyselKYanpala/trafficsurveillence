from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import*


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

"""cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)"""
cap = cv2.VideoCapture("Videos/video1.mp4")  # For Video


model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0
mask= cv2.imread("images/mask3.png")
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

pointA= [560, 250]
pointB=[724,238]
pointC=[1043,482]
pointD=[699,503]
EdgeAB=[pointB[0] - pointA[0],pointB[1] - pointA[1]]
EdgeBC=[pointC[0] - pointB[0],pointC[1] - pointB[1]]
EdgeCD=[pointD[0] - pointC[0],pointD[1] - pointC[1]]
EdgeDA=[pointA[0] - pointD[0],pointA[1] - pointD[1]]



carCount = []


while True:
    new_frame_time = time.time()
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.5:
                cvzone.cornerRect(img, (x1, y1, w, h), t=1, l=15)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1)
                currentarray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentarray))

    #cv2.line(img, (line_north[0], line_north[1]), (line_north[2], line_north[3]), (0, 0, 255), 5)
    #cv2.line(img, (line_south[0], line_south[1]), (line_south[2], line_south[3]), (0, 255, 0), 5)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    resultsTracker = tracker.update(detections)

    temp=0
    for results in resultsTracker:

        x1,y1,x2,y2,id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx=x1+w
        cy=y1+h
        AtoP = [cx- pointA[0],cy - pointA[1]]
        BtoP = [cx - pointB[0], cy - pointB[1]]
        CtoP = [cx - pointC[0], cy - pointC[1]]
        DtoP = [cx - pointD[0], cy - pointD[1]]

        crossABtoAP = (EdgeAB[0] * AtoP[1]) - (EdgeAB[1] * AtoP[0])
        crossBCtoBP = (EdgeBC[0] * BtoP[1]) - (EdgeBC[1] * BtoP[0])
        crossCDtoCP = (EdgeCD[0] * CtoP[1]) - (EdgeCD[1] * CtoP[0])
        crossDAtoDP = (EdgeDA[0] * DtoP[1]) - (EdgeDA[1] * DtoP[0])
        cvzone.cornerRect(img, (x1, y1, w, h), t=1, l=15)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        if crossABtoAP>0 and crossBCtoBP>0 and crossCDtoCP>0 and crossDAtoDP>0:
            temp = temp + 1
            """ if carCount.count(id) == 0:
                carCount.append(id)"""
        print(temp)
        cvzone.putTextRect(img, f' Count: {temp}', (50, 50))
                #cv2.line(img, (line_north[0], line_north[1]), (line_north[2], line_north[3]), (0, 255, 0), 5)"""


    #cv2.putText(img, "Car Count:6 ", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
    #cv2.putText(img, "Right Side Car Count: 3"  , (850, 100), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255),3)
    pts = np.array([[560, 250], [724,238], [1043,482], [699,503]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 0, 255),3)
    temp=0
    cv2.imshow("Image", img)
    cv2.setMouseCallback('Image', click_event)
    #cv2.imshow("mask",imgRegion)
    cv2.waitKey(1)

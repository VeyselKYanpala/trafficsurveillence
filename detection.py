from ultralytics import YOLO
import cv2
import cvzone
import math
import csv
from sort import*
import multiprocessing


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

def process_video(video_path,start_time,frame_queue):
    """cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 1280)
    cap.set(4, 720)"""

    cap = cv2.VideoCapture(video_path)  # For Video

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
    mask = cv2.imread("images/finalmask.png")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    pointA = [21, 166]
    pointB = [171, 146]
    pointC = [460, 385]
    pointD = [152, 417]
    EdgeAB = [pointB[0] - pointA[0], pointB[1] - pointA[1]]
    EdgeBC = [pointC[0] - pointB[0], pointC[1] - pointB[1]]
    EdgeCD = [pointD[0] - pointC[0], pointD[1] - pointC[1]]
    EdgeDA = [pointA[0] - pointD[0], pointA[1] - pointD[1]]

    carCount = []

    file = open(f'{video_path}.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Start Time", "End Time", "Car Count", "Crossing Count"])

    last_recorded_time = start_time
    crossing_count = 0
    frames = []
    data_list = []
    try:
        while True:
            new_frame_time = time.time()
            success, img = cap.read()

            imgRegion = cv2.bitwise_and(img, mask)
            results = model(imgRegion, stream=True)

            detections = np.empty((0, 5))
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
                            or currentClass == "motorbike" and conf > 0.7:
                        # cvzone.cornerRect(img, (x1, y1, w, h), t=1, l=15)
                        # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1)
                        currentarray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentarray))

            # cv2.line(img, (line_north[0], line_north[1]), (line_north[2], line_north[3]), (0, 0, 255), 5)
            # cv2.line(img, (line_south[0], line_south[1]), (line_south[2], line_south[3]), (0, 255, 0), 5)

            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            cvzone.putTextRect(img, f' fps: {fps}', (1860, 50))

            resultsTracker = tracker.update(detections)
            line = [(152, 417), (460, 385)]  # Çizginin iki uç noktası
            # Çizgiyi geçen araç sayısı
            temp = 0
            for results in resultsTracker:

                x1, y1, x2, y2, id = results
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx = x1 + w
                cy = y1 + h
                centerx = x1 + w // 2
                centery = y1 + h // 2
                AtoP = [cx - pointA[0], cy - pointA[1]]
                BtoP = [cx - pointB[0], cy - pointB[1]]
                CtoP = [cx - pointC[0], cy - pointC[1]]
                DtoP = [cx - pointD[0], cy - pointD[1]]

                crossABtoAP = (EdgeAB[0] * AtoP[1]) - (EdgeAB[1] * AtoP[0])
                crossBCtoBP = (EdgeBC[0] * BtoP[1]) - (EdgeBC[1] * BtoP[0])
                crossCDtoCP = (EdgeCD[0] * CtoP[1]) - (EdgeCD[1] * CtoP[0])
                crossDAtoDP = (EdgeDA[0] * DtoP[1]) - (EdgeDA[1] * DtoP[0])
                cvzone.cornerRect(img, (x1, y1, w, h), t=1, l=15)
                # cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f' fps: {fps}', (370, 50), scale=1, thickness=1)

                # Araç merkezinin çizgiyi geçip geçmediğini kontrol et
                if line[0][0] + 20 < cx < line[1][0] and line[0][1] - 20 < cy < line[1][1] + 20:
                    if id not in carCount:
                        carCount.append(id)
                        crossing_count += 1
                        print(f"Car {id} crossed the line")
                        print(f"Total cars crossed the line: {crossing_count}")

                if crossABtoAP > 0 and crossBCtoBP > 0 and crossCDtoCP > 0 and crossDAtoDP > 0:
                    temp = temp + 1
                    """ if carCount.count(id) == 0:
                        carCount.append(id)"""
                # print(temp)
                cvzone.putTextRect(img, f' Count: {temp}', (50, 50), scale=1, thickness=1)
                # cv2.line(img, (line_north[0], line_north[1]), (line_north[2], line_north[3]), (0, 255, 0), 5)"""

            if new_frame_time - last_recorded_time >= 5:  # 30 saniye geçtiyse
                car_count = temp  # temp değişkeni araç sayısını tutuyor

                # Zamanı ve araç sayısını CSV dosyasına yaz
                writer.writerow([time.strftime("%H:%M:%S", time.gmtime(last_recorded_time- start_time)),
                                 time.strftime("%H:%M:%S", time.gmtime(new_frame_time - start_time)), car_count, crossing_count])
                data_list.append([time.strftime("%H:%M:%S", time.gmtime(last_recorded_time - start_time)),
                                  time.strftime("%H:%M:%S", time.gmtime(new_frame_time - start_time)), car_count,
                                  crossing_count])
                crossing_count = 0

                last_recorded_time = new_frame_time

            # cv2.putText(img, "Car Count:6 ", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
            # cv2.putText(img, "Right Side Car Count: 3"  , (850, 100), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255),3)
            pts = np.array([pointA, pointB, pointC, pointD], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 255), 3)
            temp = 0

            crop_img = img[150:150 + 400, 540:540 + 500]
            if success:
                img = cv2.resize(img, (400, 400))  # Her bir çerçeveyi aynı boyuta getir
                frame_queue.put(img)
            # Kesilmiş görüntüyü göster
            cv2.imshow("OriginalImage", img)
            # cv2.imshow("CropImage", crop_img)
            # cv2.imshow("mask", imgRegion)
            #cv2.setMouseCallback('OriginalImage', click_event)
            cv2.waitKey(1)

    finally:
        file.close()



def combine_videos(frame_queues):
    texts = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
    while True:
        frames = [frame_queue.get() for frame_queue in frame_queues]
        for i in range(len(frames)):
            h, w, _ = frames[i].shape
            # Görüntünün alt ortasına metni çiz
            cv2.putText(frames[i], texts[i], (w//2, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        grid_row_1 = cv2.hconcat(frames[0:2])  # İlk iki videoyu yatay olarak birleştir
        grid_row_2 = cv2.hconcat(frames[2:4])  # Son iki videoyu yatay olarak birleştir
        grid = cv2.vconcat([grid_row_1, grid_row_2])  # İki satırı dikey olarak birleştir
        h, w, _ = grid.shape
        cv2.line(grid, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
        # Nihai görüntünün ortasına dikey çizgi çiz
        cv2.line(grid, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)
        cv2.imshow('Combined Video', grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def detect_objects():
    # Video yollarını bir listeye ekleyin
    video_paths = ["video_files/lane1.mp4", "video_files/lane2.mp4", "video_files/lane3.mp4", "video_files/lane4.mp4"]
    start_time = time.time()

    # Her bir video için ayrı bir işlem başlat
    frame_queues_temp = [multiprocessing.Queue() for _ in video_paths]
    processes = [multiprocessing.Process(target=process_video, args=(video_path, start_time, frame_queue)) for
                 i, (video_path, frame_queue) in enumerate(zip(video_paths, frame_queues_temp))]
    # Tüm işlemleri başlat
    for process in processes:
        process.start()
        # İşlenmiş videoları birleştirmek için bir işlem başlat
    combine_process = multiprocessing.Process(target=combine_videos, args=(frame_queues_temp,))
    combine_process.start()

    # Tüm işlemlerin tamamlanmasını bekle
    for process in processes:
        process.join()

    # combine_videos işleminin tamamlanmasını bekle
    combine_process.join()

    # İlk CSV dosyasını tamamen oku
    """dfs = [pd.read_csv(f'{video_paths[0]}.csv')]

    # Diğer CSV dosyalarını oku ve sadece ilgili sütunları seç
    for video_path in video_paths[1:]:
        dfs.append(pd.read_csv(f'{video_path}.csv').iloc[:, 2:])

    # Her bir DataFrame'in sütun adlarını güncelle
    for i, df in enumerate(dfs):
        df.columns = [f'lane{i + 1}_{col}' if i != 0 else col for col in df.columns]

    # Tüm DataFrame'leri yatay bir şekilde birleştir (axis=1)
    df_combined = pd.concat(dfs, axis=1)

    # Sonuçları bir CSV dosyasına yaz
    df_combined.to_csv('combined.csv', index=False)"""

if __name__ == "__main__":
    detect_objects()
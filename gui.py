import multiprocessing
import tkinter as tk
import time
import cv2
import pandas as pd
from PIL import Image, ImageTk
import threading
import queue


# CSV dosyalarını oku
csv_files = ["csv_files/combined.csv"]
csv_files2 = ["csv_files/scenario1.csv","csv_files/scenario2.csv",
              "csv_files/scenario3.csv","csv_files/scenario4.csv",
              "csv_files/scenario5.csv"]
csv_data = [pd.read_csv(file) for file in csv_files]
csv_data2 = [pd.read_csv(file) for file in csv_files2]
# Video dosyalarını oku
video_files = ["video_files/lane1p.mp4", "video_files/lane2p.mp4",
               "video_files/lane3p.mp4", "video_files/lane4p.mp4",
               "sumo_simulation_videos/east_west.mp4","sumo_simulation_videos/north_south.mp4",
               "sumo_simulation_videos/south_north.mp4","sumo_simulation_videos/west_east.mp4",]
cap = [cv2.VideoCapture(file) for file in video_files]

# Video frame'lerini saklamak için bir kuyruk oluştur
frame_queues = [queue.Queue(maxsize=1) for _ in cap]

# Ana tkinter penceresini oluştur
root = tk.Tk()
root.geometry("1900x1060+0+0")
for i, data in enumerate(csv_data2):
    frame = tk.Frame(root, width=80, height=30)  # Frame boyutunu küçült
    # 6'lı bir ızgara şeklinde yerleştir
    frame.grid(row=4+(i//2), column=(i%2), padx=5, pady=1, sticky='N')  # Sol alt köşeye yasla
    frame.grid_propagate(False)  # Frame'in boyutunun içerik tarafından değiştirilmesini engelle

    # Yatay kaydırma çubuğunu oluştur
    xscrollbar = tk.Scrollbar(frame, orient='horizontal')
    xscrollbar.pack(side='bottom', fill='x')

    # Kaydırma çubuğu olan bir Text widget'ı oluştur
    text = tk.Text(frame, width=30, height=7, wrap='none',  # Text boyutunu küçült
                   xscrollcommand=xscrollbar.set)
    text.pack()

    # Kaydırma çubuğunu Text widget'ına bağla
    xscrollbar.config(command=text.xview)

    text.insert('1.0', data)
# CSV dosyalarını görüntüle
for i, data in enumerate(csv_data):
    frame = tk.Frame(root, width=200, height=50)
    frame.grid(row=0, column=3, padx=5, pady=1)
    frame.grid_propagate(False)  # Frame'in boyutunun içerik tarafından değiştirilmesini engelle

    # Yatay kaydırma çubuğunu oluştur
    xscrollbar = tk.Scrollbar(frame, orient='horizontal')
    xscrollbar.pack(side='bottom', fill='x')

    # Kaydırma çubuğu olan bir Text widget'ı oluştur
    text = tk.Text(frame, width=80, height=15, wrap='none',
                   xscrollcommand=xscrollbar.set)
    text.pack()

    # Kaydırma çubuğunu Text widget'ına bağla
    xscrollbar.config(command=text.xview)

    text.insert('1.0', data)

scenarios_label = tk.Label(root, text="Scenarios", font=("Helvetica", 16))
scenarios_label.grid(row=2, column=0, padx=5, pady=1, sticky='E' )
trafficdata_label = tk.Label(root, text="Traffic Data", font=("Helvetica", 16))
trafficdata_label.grid(row=0, column=3, padx=5, pady=1, sticky='S')
results_label = tk.Label(root, text="Results", font=("Helvetica", 16))
results_label.grid(row=2, column=4, padx=5, pady=1, sticky='S')
new_csv_file = "csv_files/simulation_results.csv"
new_csv_data = pd.read_csv(new_csv_file)

# Yeni bir frame oluştur
new_frame = tk.Frame(root, width=200, height=50)
new_frame.grid(row=4, column=4, padx=5, pady=1)  # Sağ alt köşeye yasla
new_frame.grid_propagate(False)  # Frame'in boyutunun içerik tarafından değiştirilmesini engelle

# Yatay kaydırma çubuğunu oluştur
xscrollbar = tk.Scrollbar(new_frame, orient='horizontal')
xscrollbar.pack(side='bottom', fill='x')

# Text widget'ı oluştur
new_text = tk.Text(new_frame, width=40, height=7, wrap='none', xscrollcommand=xscrollbar.set)
new_text.pack()

# Kaydırma çubuğunu Text widget'ına bağla
xscrollbar.config(command=new_text.xview)
# CSV verisini Text widget'ına ekle
new_text.insert('1.0', new_csv_data)

def stream(capture, frame_queue):
    try:
        while True:
            # Video dosyasından bir frame oku
            ret, frame = capture.read()
            # Eğer video dosyasının sonuna ulaşıldıysa, başa sar
            if not ret:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Frame'i görüntülemek için PIL Image nesnesine dönüştür
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            # Görüntüyü yeniden boyutlandır
            img = img.resize((280, 280))
            imgtk = ImageTk.PhotoImage(image=img)
            # Kuyruğa yeni bir frame ekle
            if not frame_queue.full():
                frame_queue.put(imgtk)
    finally:
        capture.release()

# Her video için bir thread başlat
for capture, frame_queue in zip(cap, frame_queues):
    threading.Thread(target=stream, args=(capture, frame_queue)).start()

def update_gui(label, frame_queue):
    try:
        # Kuyruktan bir frame al
        imgtk = frame_queue.get_nowait()
        label.imgtk = imgtk
        label.configure(image=imgtk)
    except queue.Empty:
        pass
    # GUI'yi 100 milisaniye sonra tekrar güncelle
    label.after(200, update_gui, label, frame_queue)

# Video dosyalarını görüntüle
labels = [tk.Label(root) for _ in cap]
for i, label in enumerate(labels):
    if i < 4:
        label.grid(row=i // 2, column=i % 2, padx=5, pady=1, )
    else:
        label.grid(row=(i-4) % 2, column=3 + ((i - 4) % 2), padx=5, pady=1,sticky='W')
    update_gui(label, frame_queues[i])

grid_size = root.grid_size()

# Son dört widget'ın column değerini en yüksek değere ayarla
for i in range(4, 7):
    if i < 5:  # İlk iki widget (5. ve 6. videolar) en yüksek column değerine sahip olacak
        labels[i].grid(column=grid_size[0] - 1)
    else:  # Son iki widget (7. ve 8. videolar) bir sonraki column'a yerleştirilecek
        labels[i].grid(column=grid_size[0])


root.mainloop()
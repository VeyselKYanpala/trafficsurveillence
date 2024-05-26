import tkinter as tk
import cv2
import pandas as pd
from PIL import Image, ImageTk
import threading
import queue

# CSV dosyalarını oku
csv_files = ["csv_files/combined.csv"]
csv_data = [pd.read_csv(file) for file in csv_files]

# Video dosyalarını oku
video_files = ["video_files/lane1.mp4", "video_files/lane2.mp4",
               "video_files/lane3.mp4", "video_files/lane4.mp4"]
cap = [cv2.VideoCapture(file) for file in video_files]

# Video frame'lerini saklamak için bir kuyruk oluştur
frame_queues = [queue.Queue(maxsize=1) for _ in cap]

# Ana tkinter penceresini oluştur
root = tk.Tk()
root.geometry("1900x1060+0+0")

# CSV dosyalarını görüntüle
for i, data in enumerate(csv_data):
    frame = tk.Frame(root, width=400, height=200)
    frame.grid(row=0, column=4, padx=5, pady=1)
    frame.grid_propagate(False)  # Frame'in boyutunun içerik tarafından değiştirilmesini engelle

    # Yatay kaydırma çubuğunu oluştur
    xscrollbar = tk.Scrollbar(frame, orient='horizontal')
    xscrollbar.pack(side='bottom', fill='x')

    # Kaydırma çubuğu olan bir Text widget'ı oluştur
    text = tk.Text(frame, width=75, height=30, wrap='none',
                   xscrollcommand=xscrollbar.set)
    text.pack()

    # Kaydırma çubuğunu Text widget'ına bağla
    xscrollbar.config(command=text.xview)

    text.insert('1.0', data)

def stream(capture, frame_queue):
    try:
        while True:
            # Video dosyasından bir frame oku
            ret, frame = capture.read()
            if not ret:
                break
            # Frame'i görüntülemek için PIL Image nesnesine dönüştür
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            # Görüntüyü yeniden boyutlandır
            img = img.resize((200, 200))
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
    label.grid(row=i // 2, column=i % 2 + 2, padx=5, pady=1, sticky='NW')  # widget'ları sağ üst köşeye yerleştir
    update_gui(label, frame_queues[i])

root.mainloop()
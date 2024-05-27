import pandas as pd
import numpy as np


# Başlangıç ve bitiş zamanları
start_time = pd.to_datetime('00:00:00', format='%H:%M:%S')
end_time = pd.to_datetime('22:59:60', format='%H:%M:%S')

# Zaman aralıklarını oluştur
time_intervals = pd.date_range(start=start_time, end=end_time, freq='20s')
# Ensure the length of all arrays is the same
length = min(4000, len(time_intervals)-1)

# Veri setini oluştur
data = {
    'Start Time': time_intervals[:length].strftime('%H:%M:%S'),
    'End Time': time_intervals[1:length+1].strftime('%H:%M:%S'),
    'EW': np.random.randint(0, 10, size=length),
    'WE': np.random.randint(0, 10, size=length),
    'NS': np.random.randint(0, 10, size=length),
    'SN': np.random.randint(0, 10, size=length),
}

df = pd.DataFrame(data)

# Veriyi CSV dosyasına yaz
df.iloc[:4000].to_csv('csv_files/traffic_data.csv', index=False)
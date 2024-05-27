
from __future__ import absolute_import
from __future__ import print_function
import csv
import os
import sys
import random
from datetime import datetime
from sumolib import checkBinary  # noqa
import traci  # noqa
import pandas as pd

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def select_random_rows(file_name, num_rows,current_date):
    # CSV dosyasını oku
    df = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header row if there is one
        for row in reader:
            df.append(row)

    # data listesinin uzunluğunu kontrol et
    if len(df) >= num_rows:
        # rastgele bir başlangıç noktası seç
        start_index = random.randint(0, len(df) - num_rows)

        # başlangıç noktasından itibaren num_rows satır al
        sample_data = df[start_index:start_index + num_rows]

    else:
        print("Data listesi belirtilen sayıdan az öğe içeriyor.")
        sample_data = []
    """df_sample = pd.DataFrame(sample_data)
    # İlk sütunu 0'dan başlayıp 20'şer artan bir dizi ile değiştir
    df_sample.iloc[:, 0] = range(0, 20 * len(df_sample), 20)
    # İkinci sütunu, ilk sütundaki değerlere 20 ekleyerek oluştur
    df_sample.iloc[:, 1] = df_sample.iloc[:, 0] + 20
    df_sample.to_csv(f'{current_date}.csv', index=False)"""
    return sample_data
def generate_routefile(num_rows=50,time_interval=20,current_date=datetime.now().strftime("%Y-%m-%d")):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


    # Mevcut tarihi al ve yıl-ay-gün formatında bir stringe dönüştür


    # Dosya adına tarihi ekleyin
    xml_filename = f"cfg_files/maps/{current_date}.rou.xml"

    with open(xml_filename, "w") as routes:
        print("""<routes>
        <vType id="typeCar" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50" \
guiShape="passenger"/>
        <vType id="typeBus" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="EtoW" edges="EW1 EW2" />
        <route id="WtoE" edges="WE1 WE2" />
        <route id="NtoS" edges="NS1 NS2" />
        <route id="StoN" edges="SN1 SN2" />
        <route id="EtoN" edges="EW1 SN2" />
        <route id="EtoS" edges="EW1 NS2" />
        <route id="WtoN" edges="WE1 SN2" />
        <route id="WtoS" edges="WE1 NS2" />
        <route id="NtoE" edges="NS1 WE2" />
        <route id="NtoW" edges="NS1 EW2" />
        <route id="StoE" edges="SN1 WE2" />
        <route id="StoW" edges="SN1 EW2" />
        """, file=routes)

        vehNr = 0
        counter = 0
        data = []
        file_name ="csv_files/traffic_data.csv"
        sample_data = select_random_rows(file_name, num_rows,current_date)


        start_time = 0
        end_time = time_interval
        vehicles = []
        for row in sample_data:

            temp1, temp2, EW, WE, NS, SN = row
            """start_time = int(start_time.split(':')[0]) * 3600 + int(start_time.split(':')[1]) * 60 + int(
                start_time.split(':')[2])  # convert start_time to seconds
            end_time = int(end_time.split(':')[0]) * 3600 + int(end_time.split(':')[1]) * 60 + int(
                end_time.split(':')[2])  # convert end_time to seconds"""


            for i in range(int(EW)):
                depart_time = start_time + (i / int(EW)) * time_interval
                depart_time = int(depart_time)
                route = random.choice(["EtoW", "EtoN", "EtoS"])
                vehicles.append(
                    {'id': f"EW_{vehNr}", 'type': "typeCar", 'route': route, 'depart': depart_time, 'color': "1,0,0"})
                vehNr += 1

            for i in range(int(WE)):
                depart_time = start_time + (i / int(WE)) * time_interval
                depart_time = int(depart_time)
                route = random.choice(["WtoE", "WtoN", "WtoS"])
                vehicles.append(
                    {'id': f"WE_{vehNr}", 'type': "typeCar", 'route': route, 'depart': depart_time, 'color': "1,1,0"})
                vehNr += 1

            for i in range(int(NS)):
                depart_time = start_time + (i / int(NS)) * time_interval
                depart_time = int(depart_time)
                route = random.choice(["NtoS", "NtoE", "NtoW"])
                vehicles.append(
                    {'id': f"NS_{vehNr}", 'type': "typeCar", 'route': route, 'depart': depart_time, 'color': "1,0,1"})
                vehNr += 1

            for i in range(int(SN)):
                depart_time = start_time + (i / int(SN)) * time_interval
                depart_time = int(depart_time)
                route = random.choice(["StoN", "StoW", "StoE"])
                vehicles.append(
                    {'id': f"SN_{vehNr}", 'type': "typeCar", 'route': route, 'depart': depart_time, 'color': "0,0,1"})
                vehNr += 1

            start_time += time_interval
            end_time += time_interval

        # Araçları depart zamanlarına göre sıralayın
        vehicles.sort(key=lambda x: x['depart'])

        # Sıralı araçları yazdırın
        for vehicle in vehicles:
            print('    <vehicle id="{}" type="{}" route="{}" depart="{}" color="{}"/>'.format(
                vehicle['id'], vehicle['type'], vehicle['route'], vehicle['depart'], vehicle['color']), file=routes)

        print("</routes>", file=routes)



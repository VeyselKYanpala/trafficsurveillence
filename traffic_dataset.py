import csv
import random
with open("traffic.csv", mode="w",newline='') as csvfile :
    fieldnames= ["VinNorthL","totalNorthL",
                 "VinSouthL","totalSouthL",
                 "VinEastL","totalEastL",
                 "VinWestL","totalWestL",
                 "time_interval","sequence"]
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()

    vehicle_length = 4
    track_distance = 1
    vehicle_left= 6
    light_time= 15
    i=0
    start_time = 0
    totalNorthL,totalSouthL,totalEastL,totalWestL = 0,0,0,0


    while i<200:
        sequence = i % 4
        stop_time = start_time + light_time
        time_interval = str(start_time) + ":" + str(stop_time)
        VinNorthL = random.randint(0, 3)
        VinSouthL = random.randint(0, 3)
        VinEastL = random.randint(0, 3)
        VinWestL = random.randint(0, 3)

        if i % 4 == 0:
            VinNorthL -= vehicle_left
        elif i % 4 == 1:
            VinEastL -= vehicle_left
        elif i % 4 == 2:
            VinSouthL  -= vehicle_left
        else:
            VinWestL -= vehicle_left

        totalNorthL = totalNorthL + VinNorthL
        totalSouthL = totalSouthL + VinSouthL
        totalEastL = totalEastL + VinEastL
        totalWestL = totalWestL + VinWestL
        if totalNorthL <= 0:
            totalNorthL=0
        if totalSouthL <= 0:
            totalSouthL =0
        if totalEastL <= 0:
            totalEastL=0
        if totalWestL <= 0:
            totalWestL=0
        temp = dict({"VinNorthL": VinNorthL, "totalNorthL": totalNorthL,
                     "VinSouthL": VinSouthL, "totalSouthL": totalSouthL,
                     "VinEastL": VinEastL, "totalEastL": totalEastL,
                     "VinWestL": VinWestL, "totalWestL": totalWestL,
                     "time_interval": time_interval, "sequence": sequence})
        start_time= stop_time


        writer.writerow(temp)
        i+=1
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import sumocfg_generator
import xml_to_stat
from csv_to_xml import generate_routefile
import model_train
import csv
import os
import time
def generate():
    sim_nu = 5
    currents_dates = []

    for i in range(sim_nu):
        current_date = datetime.now().strftime('%d-%H-%M-%S') + f"{i}"
        generate_routefile(50, 20, current_date)
        sumocfg_generator.generate_cfg_file(current_date)
        currents_dates.append(current_date)
    return currents_dates

def main():
    cfg_names = generate()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(model_train.run, False, "model_11", 50, 1500, cfg_file) for cfg_file in cfg_names]

    # Wait for all processes to complete and get their results
    results = [future.result() for future in futures]
    results_with_names = zip(cfg_names, results)

    # Eşleşmeleri bir CSV dosyasına yaz
    with open('csv_files/simulation_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Simulation Name", "AVG Fuel", "AVG CO2", "AVG Waiting Time"])  # Sütun başlıklarını yaz
        for name, result in results_with_names:
            writer.writerow([name] + list(result))

    return results,cfg_names



if __name__ == '__main__':
    main()


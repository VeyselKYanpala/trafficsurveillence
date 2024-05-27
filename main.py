from datetime import datetime
from multiprocessing import Process
import sumocfg_generator
import xml_to_stat
from csv_to_xml import generate_routefile
import model_train
import os
import time
def generate():
    sim_nu = 5
    cfg_files = []


    for i in range(sim_nu):
        current_date = datetime.now().strftime('%d-%H-%M-%S') + f"{i}"
        generate_routefile(50, 20, current_date)
        sumocfg_generator.generate_cfg_file(current_date)
        cfg_files.append(current_date)
        print(cfg_files)
    return cfg_files
def main():
    processes = []
    cfg_files = generate()


    for cfg_file in cfg_files:
        # Run model_train.run function in a separate process
        print(cfg_file)
        process = Process(target=model_train.run, args=(False, "model_11", 50, 1500, cfg_file))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()

        # Tüm süreçlerin tamamlanmasını bekle


if __name__ == '__main__':
    main()


import xml.etree.ElementTree as ET

def calculate_averages(file_path):
    # XML dosyasını oku
    tree = ET.parse(file_path)
    root = tree.getroot()

    # CO2 emisyon değerlerini, bekleme sürelerini ve yakıt tüketim değerlerini saklamak için listeler oluştur
    co2_emissions = []
    waiting_times = []
    fuel_abs_values = []

    # XML dosyasındaki her 'tripinfo' öğesini döngüye al
    for tripinfo in root.findall('tripinfo'):
        # 'emissions' öğesini bul ve 'CO2_abs' özniteliğini al
        emissions = tripinfo.find('emissions')
        co2_emission = float(emissions.get('CO2_abs'))
        co2_emissions.append(co2_emission)

        # 'waitingTime' özniteliğini al
        waiting_time = float(tripinfo.get('waitingTime'))
        waiting_times.append(waiting_time)

        # 'fuel_abs' özniteliğini al
        fuel_abs = float(emissions.get('fuel_abs'))
        fuel_abs_values.append(fuel_abs)

    # CO2 emisyon değerlerinin, bekleme sürelerinin ve yakıt tüketim değerlerinin ortalama değerlerini hesapla
    average_co2_emission = sum(co2_emissions) / len(co2_emissions)
    average_waiting_time = sum(waiting_times) / len(waiting_times)
    average_fuel_abs = sum(fuel_abs_values) / len(fuel_abs_values)

    return average_co2_emission, average_waiting_time, average_fuel_abs
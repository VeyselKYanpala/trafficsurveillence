import xml.etree.ElementTree as ET


def generate_cfg_file( current_date):
    # XML ağacını oluştur
    configuration = ET.Element('configuration', {
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/sumoConfiguration.xsd'
    })

    # 'input' elementini oluştur ve 'configuration' elementine ekleyin
    input = ET.SubElement(configuration, 'input')
    ET.SubElement(input, 'net-file', {'value': 'maps/crosnew.net.xml'})
    ET.SubElement(input, 'route-files', {'value': f'maps/{current_date}.rou.xml'})

    # 'output' elementini oluştur ve 'configuration' elementine ekleyin
    output = ET.SubElement(configuration, 'output')
    ET.SubElement(output, 'emission-output', {'value': f'outputs/{current_date}_emissions.xml'})
    ET.SubElement(output, 'queue-output', {'value': f'outputs/{current_date}_queue.xml'})
    ET.SubElement(output, 'tripinfo-output', {'value': f'outputs/{current_date}.xml'})

    # 'time' elementini oluştur ve 'configuration' elementine ekleyin
    time = ET.SubElement(configuration, 'time')
    ET.SubElement(time, 'begin', {'value': '0'})

    # 'report' elementini oluştur ve 'configuration' elementine ekleyin
    report = ET.SubElement(configuration, 'report')
    ET.SubElement(report, 'verbose', {'value': 'true'})
    ET.SubElement(report, 'no-step-log', {'value': 'true'})

    # XML ağacını bir stringe dönüştür
    xml_str = ET.tostring(configuration, encoding='utf-8', method='xml')

    # XML stringini düzgün bir şekilde biçimlendir
    xml_str = xml_str.decode().replace('><', '>\n<')

    # XML stringini dosyaya yaz
    with open(f'cfg_files/{current_date}.sumocfg', 'w') as f:
        f.write(xml_str)
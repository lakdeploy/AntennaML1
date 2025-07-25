import os
from xml.etree.ElementTree import Element, SubElement, ElementTree

def export_to_hfss_xml(config_dict, filename="hfss_export.xml"):
    os.makedirs("exports", exist_ok=True)
    root = Element('HFSSDesign')
    params = SubElement(root, 'Parameters')
    for key, value in config_dict.items():
        param = SubElement(params, 'Parameter',
                           name=key, value=str(value),
                           unit=guess_unit_from_name(key))
    export_path = os.path.join("exports", filename)
    tree = ElementTree(root)
    tree.write(export_path, encoding="utf-8", xml_declaration=True)
    return export_path

def guess_unit_from_name(param):
    # same as before
    if "freq" in param.lower(): return "GHz"
    if "gain" in param.lower(): return "dB"
    if "power" in param.lower(): return "dBm"
    if "density" in param.lower(): return "users/km^2"
    if "speed" in param.lower(): return "km/h"
    if "interference" in param.lower(): return "ratio"
    return "mm"

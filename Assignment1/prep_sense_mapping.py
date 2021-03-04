"""
@author: Tezan Sahu
"""

import pandas as pd
import os
import re
import json
import xml.etree.ElementTree as ET

tpp_mappings_df = pd.read_csv("./data/Tratz Variations/mappings_for_tpp_data", sep="\t", header=None, names=["id", "prep_sense"])
tpp_mappings = tpp_mappings_df.set_index('id').T.to_dict('records')[0]

prep_mappings = {}

data_dir = "./data/Train/Source"

for prep_file in os.listdir(data_dir):
    preposition = re.findall(r"pp-([a-z]*)\.", prep_file)[0]
    print(preposition)
    prep_raw_senses = []
    prep_senses = []

    tree = ET.parse(os.path.join(data_dir, prep_file)) 
    root = tree.getroot()

    for item in root.findall('./instance'):
        try:
            instance_id = item.attrib["id"]
            raw_sense = item.find("./answer").attrib["senseid"]
            sense = tpp_mappings[instance_id]
            
            prep_raw_senses.append(raw_sense)
            prep_senses.append(sense)

        except Exception:
            pass

    df = pd.DataFrame({"raw_senseid": prep_raw_senses, "tpp_senseid": prep_senses})
    df = df.drop_duplicates().reset_index(drop=True)
    prep_mappings[preposition] = df.set_index('raw_senseid').T.to_dict('records')[0]

with open("sense_mapping.json", "w") as f:
    f.write(json.dumps(prep_mappings, indent=4))

import os
import json

# 转换labelocr的txt标注结果为SER训练要用的train.json
from typing import List

org_path = "../train_data/ssalg/train.json"

id = 0
labels=set()

with open(org_path, 'r', encoding='utf-8') as source_file:
    for line in source_file:
        str_list: List[str] = line.split('\t')
        file_name = str_list[0]
        json_str: object = json.loads(str_list[1])

        for item in json_str:
            labels.add(item['label'])

    print(labels)
    print(len(labels))

import os
import json

##### 手动把第一列的路径+文件名改为文件名，即去掉前缀路径 #####
# 转换labelocr的txt标注结果为SER训练要用的train.json
from typing import List

### val同 train，即把train.txt改为val.txt #####
org_path = "../train_data/ssalg/train.txt"
dist_path = "../train_data/ssalg/train.json"

id = 0

with open(org_path, 'r', encoding='utf-8') as source_file:
    with open(dist_path, 'w', encoding='utf-8') as dest_file:
        for line in source_file:
            print('----'*10)
            print(line)

            str_list: List[str] = line.split('\t')
            file_name = str_list[0]
            json_str: object = json.loads(str_list[1])

            for item in json_str:
                item['id'] = id
                id += 1
                del item['difficult']
                item['label'] = item.pop('key_cls')
                item['linking'] = ((id-1,id),)

                if item['label'] == 'None':
                    item['label'] = 'QIT'

            res = json.dumps(json_str, ensure_ascii=False)

            print(res)
            print('*****' * 10)
            # write
            dest_file.write(file_name + '\t' + res +'\n')

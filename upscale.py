import numpy as np
import json
import os
import os.path
import pandas as pd


def get_inference_json(jsons_path):
    output = []
    print(jsons_path)
    for path in os.listdir(jsons_path):
        if os.path.isfile(os.path.join(jsons_path, path)) and path.split(".")[0][-1] == "r":
        #if os.path.exists(os.path.join(jsons_path,("sg2im_"+str(screen['id'])+".json"))):
            f = open(os.path.join(jsons_path, path))
            #f = open(os.path.join(jsons_path,("sg2im_"+str(screen['id'])+".json")))
            temp = json.load(f)
            f.close()
            for i in range(len(temp['objects'])):
                if temp['objects'][i]['box'] != None:
                    temp['objects'][i]['box'][0] = round(temp['objects'][i]['box'][0]*18/720)*40
                    temp['objects'][i]['box'][1] = round(temp['objects'][i]['box'][1]*32/1280)*40
                    temp['objects'][i]['box'][2] = round(temp['objects'][i]['box'][2]*18/720)*40
                    temp['objects'][i]['box'][3] = round(temp['objects'][i]['box'][3]*32/1280)*40
                    if temp['objects'][i]['box'][2] == temp['objects'][i]['box'][0]:
                        temp['objects'][i]['box'][2] += 18
                    if temp['objects'][i]['box'][3] == temp['objects'][i]['box'][1]:
                        temp['objects'][i]['box'][3] += 32
                    if temp['objects'][i]['class'] == "ROOT":
                        temp['objects'][i]['box'][0] = 0
                        temp['objects'][i]['box'][1] = 0
                        temp['objects'][i]['box'][2] = 720
                        temp['objects'][i]['box'][3] = 1280
            output.append(temp)
    return output

if __name__ == '__main__':
    out = get_inference_json("experiments/clay_seq2seq_upsample_manual/test_1/sg2im_json")
    with open('upsample.json', 'w') as json_file:
        json.dump(out, json_file)
import json
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



def main():
    train_path = "train.json"
    test_path = "test.json"
    f = open(train_path)
    data = json.load(f)
    f.close()
    cell = [5,5]
    for s in range(len(data)):
        for o in range(len(data[s]["objects"])):
            data[s]["objects"][o]["box"][0] = round(data[s]["objects"][o]["box"][0]/cell[0])
            data[s]["objects"][o]["box"][1] = round(data[s]["objects"][o]["box"][1]/cell[1])
            data[s]["objects"][o]["box"][2] = round(data[s]["objects"][o]["box"][2]/cell[0])
            data[s]["objects"][o]["box"][3] = round(data[s]["objects"][o]["box"][3]/cell[1])
    train_path2 = "train_grid_5.json"
    json_object = json.dumps(data, indent = 4)
    with open(train_path2, "w") as outfile:
        outfile.write(json_object)
    
    f = open(test_path)
    data = json.load(f)
    f.close()
    cell = [5,5]
    for s in range(len(data)):
        for o in range(len(data[s]["objects"])):
            data[s]["objects"][o]["box"][0] = round(data[s]["objects"][o]["box"][0]/cell[0])
            data[s]["objects"][o]["box"][1] = round(data[s]["objects"][o]["box"][1]/cell[1])
            data[s]["objects"][o]["box"][2] = round(data[s]["objects"][o]["box"][2]/cell[0])
            data[s]["objects"][o]["box"][3] = round(data[s]["objects"][o]["box"][3]/cell[1])
    test_path2 = "test_grid_5.json"
    json_object = json.dumps(data, indent = 4)
    with open(test_path2, "w") as outfile:
        outfile.write(json_object)



if __name__ == "__main__":
	main()
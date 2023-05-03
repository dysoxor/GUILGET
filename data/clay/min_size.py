import json
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



def main():
    train_path = "train.json"
    f = open(train_path)
    data = json.load(f)
    min_x = 1440
    min_y = 2560
    num_of_screens = 0
    num_of_comp = 0
    for screen in data:
        is_less = False
        for obj in screen["objects"]:
            min_x = min(min_x, obj['box'][2]-obj['box'][0])
            min_y = min(min_y, obj['box'][3]-obj['box'][1])

            if(obj['box'][2]-obj['box'][0]) <= 2 or (obj['box'][3]-obj['box'][1]) <= 2:
                num_of_comp += 1
                is_less = True
                #print(screen["objects"])
        if is_less:
            num_of_screens += 1
    print(min_x, min_y, num_of_screens, num_of_comp)



if __name__ == "__main__":
	main()
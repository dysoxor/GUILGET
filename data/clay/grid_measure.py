
import json
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



def main():
    train_path = "train.json"
    test_path = "test.json"
    num_of_cols = 4
    d = 60
    x = []
    y = []
    f = open(train_path)
    data = json.load(f)
    for di in range(d):
        gutters = (di+1)/360
        col_size = (360 - 5*(di+1))/(4*360)
        cols_y = [0, gutters]
        for c in range(num_of_cols):
            cols_y.append(cols_y[-1] + col_size)
            cols_y.append(cols_y[-1] + gutters)
        
        tot_loss = 0
        N = 0
        for screen in data:
            for obj in screen["objects"]:
                min_dist = 999
                for c in range(len(cols_y)//2):
                    min_dist = min(min_dist, abs(cols_y[2*c +1] - obj['box'][0]/screen['width']))
                    min_dist = min(min_dist, abs(cols_y[2*c] - obj['box'][0]/screen['width']))
                    min_dist = min(min_dist, abs(cols_y[2*c] - obj['box'][2]/screen['width']))
                    min_dist = min(min_dist, abs(cols_y[2*c + 1] - obj['box'][2]/screen['width']))
                tot_loss += min_dist
                N += 1
        x.append(di+1)
        y.append(tot_loss/N)
    # plotting the points 
    plt.plot(x, y)
    
    # naming the x axis
    plt.xlabel('glutter size %')
    # naming the y axis
    plt.ylabel('average distance to a glutter edge')
    
    # giving a title to my graph
    #plt.title('My first graph!')
    
    # function to show the plot
    plt.savefig("glutter2.png")



if __name__ == "__main__":
	main()
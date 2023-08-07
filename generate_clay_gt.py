from bounding_box import bounding_box as bb
import json
import numpy as np
import os
import cv2

label_map = {"ROOT":0,
"BACKGROUND":1,
"IMAGE":2,
"PICTOGRAM":3,
"BUTTON":4,
"TEXT":5,
"LABEL":6,
"TEXT INPUT":7,
"MAP":8,
"CHECK BOX":9,
"SWITCH":10,
"PAGER INDICATOR":11,
"SLIDER":12,
"RADIO BUTTON":13,
"SPINNER":14,
"PROGRESS BAR":15,
"ADVERTISEMENT":16,
"DRAWER":17,
"NAVIGATION BAR":18,
"TOOLBAR":19,
"LIST ITEM":20,
"CARD VIEW":21,
"CONTAINER":22,
"DATE PICKER":23,
"NUMBER STEPPER":24}


def xyxy2xywh(box, size_original, size_target):
    box[0] = (box[0]/size_original[1])*size_target[1]
    box[1] = (box[1]/size_original[0])*size_target[0]
    box[2] = (box[2]/size_original[1])*size_target[1]
    box[3] = (box[3]/size_original[0])*size_target[0]
    #box[2] = box[2] - box[0]
    #box[3] = box[3] - box[1]
    return box
def xywh_norm(box, size_original, size_target):
    box[0] = (box[0]/size_original[1])*size_target[1]
    box[1] = (box[1]/size_original[0])*size_target[0]
    box[2] = (box[2]/size_original[1])*size_target[1]
    box[3] = (box[3]/size_original[0])*size_target[0]
    return box

def show_and_save(image, path):
    cv2.imwrite(path, image)

def draw_img(image_size_original, image_size_target, boxes, labels, label_ids, save_dir, name):
    # setting color
    color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
            'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    image = np.full(image_size_target, 200.)
    #print(image.shape)
    boxes[boxes < 0] = 1
    boxes[boxes > image_size_target[0]] = image_size_target[0] - 1
    for i in range(len(boxes)):
        bb.add(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                str(labels[i]+'[{}]'.format(label_ids[i])), 
                color=color[ord(labels[i][0])-ord('a')])
    show_and_save(image,os.path.join(save_dir,str(name) + '_{}'.format(len(boxes))  +'.png'))

def draw_img_segmentation(image_size_original, image_size_target, boxes, labels, label_ids, save_dir, name):
        # setting color
        color = [(125, 112, 186), (242, 242, 62), (50, 99, 108), (167, 161, 189), (145, 75, 94), (47, 225, 162), (105, 85, 77), (86, 15, 119), 
                 (186, 255, 222), (73, 77, 143), (37, 0, 116), (45, 184, 16), (209, 172, 222), (9, 212, 166), (50, 90, 250) ,(227, 127, 181),
                (195, 182, 25), (247, 191, 42), (57, 131, 94), (160, 243, 132), (120, 64, 240), (180, 63, 107), (208, 8, 164), (70, 129, 159), 
                 (194, 251, 210)]
        image = np.zeros(image_size_target,'uint8')
        image.fill(255)
        boxes[boxes < 0] = 1
        boxes[boxes > image_size_target[0]] = image_size_target[0] - 1
        drawn = []
        order = []
        #print("start drawing")
        for i in reversed(range(len(boxes))):
            #print("first loop ", i)
            if label_ids[i] != 0 and label_ids[i] not in drawn:
                drawn.append(label_ids[i])
                #print("append drawn ",label_ids[i])
                #print("order before ", order)
                if len(order) > 0:
                    has_changed = False
                    for j in range(len(order)):
                        if boxes[i][2]-boxes[i][0] + boxes[i][3]-boxes[i][1] >= boxes[order[j]][2]-boxes[order[j]][0] + boxes[order[j]][3]-boxes[order[j]][1]:
                            order.insert(j, i)
                            has_changed = True
                            break
                    if not has_changed:
                        order.append(i)
                else:
                    order.append(i)
                #print("order after ", order)
        for i in order:
            #print("second loop ", i)
            image = cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),
                color[label_map[labels[i]]], -1)
        #print("saving")
        show_and_save(image,
                   os.path.join(save_dir,str(name)+'.png'))
        

def draw_specific_screen(image_size_original, image_size_target, components, save_dir, name):
    color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
            'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
    image = np.full(image_size_target, 200.)
    #print(image.shape)
    boxes = []
    labels = []
    for c in components:
        boxes.append(xywh_norm(list(c[1]), image_size_original, image_size_target))
        labels.append(c[0])
    boxes = np.array(boxes)
    boxes[boxes < 0] = 1
    boxes[boxes > image_size_target[0]] = image_size_target[0] - 1
    for i in range(len(boxes)):
        bb.add(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                str(labels[i]), 
                color=color[ord(labels[i][0])-ord('a')])
    show_and_save(image,os.path.join(save_dir,str(name) + '_{}'.format(len(boxes))  +'.png'))


def load_objects(file_path):
    f = open(file_path)
    data = json.load(f)

    for screen in data:
        #print("screen id ", screen["id"])
        labels = []
        label_idx = []
        boxes = []
        name = screen["id"]
        image_size_original = (2560, 1440, 3)
        image_size_target = (960, 540, 3)
        for obj in screen["objects"]:
                
            if obj["class"] != None:
                #if obj["box"][0] < 0:
                #    obj["box"][0] = 0
                #if obj["box"][1] < 0:
                #    obj["box"][1] = 0
                #if obj["box"][2] > 720:
                #    obj["box"][2] = 720
                #if obj["box"][3] > 1280:
                #    obj["box"][3] = 1280
                labels.append(obj["class"])
                label_idx.append(len(label_idx)+1)
                boxes.append(xyxy2xywh(obj["box"], image_size_original, image_size_target))
        boxes_np = np.array(boxes)
        #print(boxes, labels, label_idx)
        draw_img_segmentation(image_size_original, image_size_target, boxes_np, labels, label_idx, "./data/clay/layout_test", name)




def main():
    train_path = "upsample.json"
    test_path = "/home/travail/ansoba/GUILGET/data/clay/test.json"
    #components = [('ROOT', [0, 0, 1440, 2560], None, 0), ('CONTAINER_0', (0, 280, 1440, 2392), {'box': [0, 0, 1440, 2560], 'class': 'ROOT'}, 1), ('LIST ITEM_0', (0, 280, 1440, 1927), {'box': (0, 280, 1440, 2392), 'class': 'CONTAINER_0'}, 2), ('TEXT_0', (56, 1645, 1384, 1899), {'box': (0, 280, 1440, 1927), 'class': 'LIST ITEM_0'}, 3), ('CONTAINER_1', (56, 1363, 1384, 1617), {'box': (0, 280, 1440, 1927), 'class': 'LIST ITEM_0'}, 3), ('CONTAINER_2', (0, 1122, 1440, 1321), {'box': (0, 280, 1440, 1927), 'class': 'LIST ITEM_0'}, 3), ('TOOLBAR_0', (0, 84, 1440, 280), {'box': [0, 0, 1440, 2560], 'class': 'ROOT'}, 1), ('CONTAINER_3', (0, 926, 1440, 1094), {'box': (0, 280, 1440, 1927), 'class': 'LIST ITEM_0'}, 3), ('CONTAINER_4', (0, 1927, 1440, 2093), {'box': (0, 280, 1440, 2392), 'class': 'CONTAINER_0'}, 2), ('PICTOGRAM_0', (482, 364, 958, 840), {'box': (0, 280, 1440, 1927), 'class': 'LIST ITEM_0'}, 3), ('LABEL_0', (56, 1164, 1384, 1321), {'box': (0, 1122, 1440, 1321), 'class': 'CONTAINER_2'}, 4), ('TEXT_1', (70, 1447, 1370, 1533), {'box': (56, 1363, 1384, 1617), 'class': 'CONTAINER_1'}, 4), ('TEXT_2', (734, 940, 1384, 1080), {'box': (0, 926, 1440, 1094), 'class': 'CONTAINER_3'}, 4), ('TEXT_3', (56, 940, 706, 1080), {'box': (0, 926, 1440, 1094), 'class': 'CONTAINER_3'}, 4), ('TEXT_4', (837, 1955, 1384, 2040), {'box': (0, 1927, 1440, 2093), 'class': 'CONTAINER_4'}, 3), ('PICTOGRAM_1', (0, 84, 196, 280), {'box': (0, 84, 1440, 280), 'class': 'TOOLBAR_0'}, 2), ('TEXT_5', (511, 139, 929, 225), {'box': (0, 84, 1440, 280), 'class': 'TOOLBAR_0'}, 2), ('PICTOGRAM_2', (1272, 98, 1440, 266), {'box': (0, 84, 1440, 280), 'class': 'TOOLBAR_0'}, 2)]

    #draw_specific_screen((2560, 1440, 3), (640, 640, 3), components, "../data/clay", "68068")
    load_objects(test_path)
    



if __name__ == "__main__":
	main()

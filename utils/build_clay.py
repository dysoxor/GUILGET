import numpy as np
#import cv2
import matplotlib.pyplot as plt
import json
import random
#import networkx as nx
import os
import pandas as pd
import os.path
# %matplotlib inline
import operator
label_map = {
	-1:"ROOT",
	0:"BACKGROUND",
	1:"IMAGE",
	2:"PICTOGRAM",
	3:"BUTTON",
	4:"TEXT",
	5:"LABEL",
	6:"TEXT INPUT",
	7:"MAP",
	8:"CHECK BOX",
	9:"SWITCH",
	10:"PAGER INDICATOR",
	11:"SLIDER",
	12:"RADIO BUTTON",
	13:"SPINNER",
	14:"PROGRESS BAR",
	15:"ADVERTISEMENT",
	16:"DRAWER",
	17:"NAVIGATION BAR",
	18:"TOOLBAR",
	19:"LIST ITEM",
	20:"CARD VIEW",
	21:"CONTAINER",
	22:"DATE PICKER",
	23:"NUMBER STEPPER"
}
opposite_relations = {
    "right": "left",
    "left": "right",
    "below": "above",
    "above": "below"
}

def bbox_conv(box):
    """convert bounding boxes from xywh to x0y0x1y1

    Args:
        box (list(int)): bouding box

    Returns:
        list(int): converted bounding box
    """
    x0 = box[0]
    y0 = box[1]
    x1 = box[0]+box[2]
    y1 = box[1]+box[3]
    return [x0,y0,x1,y1]

def check_inside(box1,box2):
    """check whether box1 is contained in box2

    Args:
        box1 (list(int)): bounding box
        box2 (list(int)): bounding box
    Returns:
        Boolean: whether box1 is contained in box2
    """
    conv1 = box1
    conv2 = box2
    if(conv1[0] < conv2[0] or conv1[1] < conv2[1] or conv1[2] > conv2[2] or conv1[3] > conv2[3]):
        return False
    return True

def wh_conv(box):
    """
    compute the width and height of a box
    Args:
        box1: tuple(int): bounding box
    Returns:
        tuple(int): width and height
    """
    return (box[2]-box[0],box[3]-box[1])

def Traversal(data):
    """Helper that traverse the json file returns a dictionary with data as key and area as value"""
    item_dict = {}
    if ("children" not in data):
        return
    queue = []
    for j in data['children']:
        if j != None:
            queue.append(j)
    #queue.append(data)
    while(len(queue) > 0):
        node = queue.pop()
        bounds = node["bounds"]
        node1 = {"bounds": tuple(node["bounds"]),"pointer": node["pointer"]}
        frozen_node = frozenset(node1.items())
        #item_dict[frozen_node] = bounds[2] * bounds[3]
        wh = wh_conv(bounds)
        item_dict[frozen_node] = wh[0] * wh[1]
        if("children" not in node):
            continue
        for i in node['children']:
            if i != None:
                queue.append(i)
    return item_dict

def parse_json(data, df, components, items_dico, objects, ind=1, parent={"box": [0, 0, 1440, 2560], "class": "ROOT"}):
    """parse a json file from Rico

    Args:
        data (dict): json file to parse
        df (dataframe): csv file from clay (clay_labels.csv) of the specific screen_id
        components(list(tuples)): tuples are composed of (unique name of the component, bounding box, parent, depth in the json)
        items_dico(dict): count number of layouts within this data (json file)
        objects(list(dict)): dictionaries are composed of {"box": bounding box, "class": component name, "parent": parent id}
        ind (int): depth in the json
        parent (dict): parent component composed of {"box": bounding box of the parent, "class": name of the parent}

    Returns:
        components (list(tuples))
        items_dico (dict)
        objects (list(dict))
    """
    #components and objects already include root

    #a item dictionary storing each item and its area
    # i is next data

    #item_dict[data] = 1440*2560
    #print("wrong")
    node = df['node_id'].values  # return an array of list containing node_id
    data_store = data
    item_dict = Traversal(data_store)
    sorted_item_d = dict(sorted(item_dict.items(), key=operator.itemgetter(1), reverse=True))
    pointer = 0
    for key in list(sorted_item_d):
        key2 = dict(key)
        # use list for python 3.x use .keys for 2.x
        #pointer = 0
        Is_under_root = True
        #grabbing layout
        layout = None
        if key2 != None and key2["pointer"] in node:
            # check if there is a child and whether its pointer and is in clay_label.csv[node_id] of the specific screen_id
            layout = label_map[df.loc[df['node_id'] == key2["pointer"]]["label"].values.tolist()[0]]
        if layout == None:
            del sorted_item_d[key]
            continue
        else:
            for j in range(pointer-1,-1,-1):
                j_key = list(sorted_item_d)[j]
                j_key = dict(j_key)
                #print(j_key["bounds"])
                if(check_inside(key2["bounds"],j_key["bounds"])):
                    p = {"box": components[j+1][1], "class": components[j+1][0]}
                    p = components[j+1][0]
                    #p = components[j + 1][2]
                    jind = components[j + 1][3]
                    if layout not in items_dico and (key2["bounds"][2] - key2["bounds"][0]) > 0 and (key2["bounds"][3] - key2["bounds"][1]) > 0:
                        items_dico[layout] = 1
                        components.append((layout + "_0", key2["bounds"],p ,jind + 1))
                        objects.append({"box": key2["bounds"], "class": layout.replace("_", " "), "parent": 0})
                    elif (key2["bounds"][2] - key2["bounds"][0]) > 0 and (key2["bounds"][3] - key2["bounds"][1]) > 0:
                        components.append((layout + "_" + str(items_dico[layout]), key2["bounds"], p ,jind + 1))
                        objects.append({"box": key2["bounds"], "class": layout.replace("_", " "), "parent": 0})
                        items_dico[layout] = items_dico[layout] + 1
                    Is_under_root = False #not a children of the root but children of j_key
                    break
            # for child of root
            if(Is_under_root):
                if layout not in items_dico and (key2["bounds"][2] - key2["bounds"][0]) > 0 and (key2["bounds"][3] - key2["bounds"][1]) > 0:
                    items_dico[layout] = 1
                    components.append((layout + "_0", key2["bounds"], parent["class"], ind))
                    objects.append({"box": key2["bounds"], "class": layout.replace("_", " "), "parent": 0})
                elif (key2["bounds"][2] - key2["bounds"][0]) > 0 and (key2["bounds"][3] - key2["bounds"][1]) > 0:
                    components.append((layout + "_" + str(items_dico[layout]), key2["bounds"], parent["class"], ind))
                    objects.append({"box": key2["bounds"], "class": layout.replace("_", " "), "parent": 0})
                    items_dico[layout] = items_dico[layout] + 1
        pointer += 1
    return (components, items_dico, objects)


def parse_json_old(data, df, components, items_dico, objects, ind=1, parent={"box": [0,0,1440,2560], "class": "ROOT"}):
    """parse a json file from Rico
    
    Args:
        data (dict): json file to parse
        df (dataframe): csv file from clay (clay_labels.csv)
        components(list(tuples)): tuples are composed of (unique name of the component, bounding box, parent, depth in the json)
        items_dico(dict): used to give unique names to components
        objects(list(dict)): dictionaries are composed of {"box": bounding box, "class": component name, "parent": parent id}
        ind (int): depth in the json
        parent (dict): parent component composed of {"box": bounding box of the parent, "class": name of the parent}
        
    Returns:
        components (list(tuples))
        items_dico (dict)
        objects (list(dict))
    """

    node = df['node_id'].values
    for i in data['children']:
        layout = None
        if i != None and i["pointer"] in node:
            layout = label_map[df.loc[df['node_id'] == i["pointer"]]["label"].values.tolist()[0]]
		#layout = i["componentLabel"]
		#print(2*(ind+1)*"-",layout, " (", i["pointer"], ",", node, ")")
        if i != None and "children" in i:
            if layout != None:
                if layout not in items_dico:
                    items_dico[layout]=1
                    components.append((layout+"_0", i["bounds"],parent['class'], ind))
                    objects.append({"box": i["bounds"], "class":layout.replace("_", " "), "parent": 0})
                else:
                    components.append((layout+"_"+str(items_dico[layout]), i["bounds"], parent['class'], ind))
                    objects.append({"box": i["bounds"], "class":layout.replace("_", " "), "parent": 0})
                    items_dico[layout] = items_dico[layout]+1
            if len(components) == 0:
                p = None
            else:
                p= {"box": components[-1][1],"class": components[-1][0]}
            (components, items_dico, objects) = parse_json_old(data=i, df=df, ind=ind+1, parent=p, components=components, items_dico=items_dico, objects=objects)
        elif layout != None:
            if layout not in items_dico:
                items_dico[layout]=1
                components.append((layout+"_0", i["bounds"], parent['class'], ind))
                objects.append({"box": i["bounds"], "class":layout.replace("_", " "), "parent": 0})
            else:
                components.append((layout+"_"+str(items_dico[layout]), i["bounds"], parent['class'], ind))
                objects.append({"box": i["bounds"], "class":layout.replace("_", " "), "parent": 0})
                items_dico[layout] = items_dico[layout]+1
    return (components, items_dico, objects)

def get_center(box):
    """return center of the bounding box

    Args:
        box (list(int)): bounding box

    Returns:
        tuple(int,int): coordinates of the center
    """
    #width = box[2]-box[0]
    #height = box[3] - box[1]
    #x = box[0]+(width/2)
    #y = box[1] + (height/2)
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)

def get_relation(box1, box2):
    """return the relation between box1 and box2

    Args:
        box1 (list(int)): bounding box
        box2 (list(int)): bounding box

    Returns:
        string: relation
    """
    bb1 = [i for i in box1]
    bb2 = [i for i in box2]
    
    bb1[0] = max(0, bb1[0])
    bb1[1] = max(0, bb1[1])
    bb1[2] = min(1440, bb1[2])
    bb1[3] = min(2560, bb1[3])
    
    bb2[0] = max(0, bb2[0])
    bb2[1] = max(0, bb2[1])
    bb2[2] = min(1440, bb2[2])
    bb2[3] = min(2560, bb2[3])
    center1 = get_center(bb1)
    center2 = get_center(bb2)
    
    right = max(-1, bb1[0] - bb2[2])
    left = max(-1, bb2[0] - bb1[2])
    below = max(-1, bb1[1] - bb2[3])
    above = max(-1, bb2[1] - bb1[3])
    
    res = max(right, left)
    res = max(res, below)
    res = max(res, above)
    
    if res == -1:
        right = max(-1, bb1[2] - bb2[2])
        left = max(-1, bb2[0] - bb1[0])
        below = max(-1, bb1[3] - bb2[3])
        above = max(-1, bb2[1] - bb1[1])
        
        res = max(right, left)
        res = max(res, below)
        res = max(res, above)
    
    if res == right:
        return "right"
    elif res == left:
        return "left"
    elif res == below:
        return "below"
    else:
        return "above"
    
    """if abs(center1[0] - center2[0]) > abs(center1[1] - center2[1]):
        if center1[0] > center2[0]:
            return "right"
        else:
            return "left"
    else:
        if center1[1] > center2[1]:
            return "below"
        else:
            return "above"
    """
def check_coverage(components):
    """are the bounding boxes covering at least 1/4 of the screen

    Args:
        components (list): components extracted from json

    Returns:
        Bool: coverage check
    """

    covered_surface = 0
    total_surface = 1440*2560
    min_i = 999999
    min_box = [999999, 999999, -10, -10]
    for c in components:
        if c[3] != 0 and c[3] <= min_i and c[0] != 'ROOT':
            if c[3] < min_i:
                min_i = c[3]
                covered_surface = (c[1][2]*c[1][3])
            else:
                covered_surface += (c[1][2]*c[1][3])
        #if c[3] != 0:
        #    if c[1][0] < min_box[0]:
        #        min_box[0] = c[1][0]
        #    if c[1][1] < min_box[1]:
        #        min_box[1] = c[1][1]
        #    if c[1][0]+c[1][2] > min_box[2]:
        #        min_box[2] = c[1][0]+c[1][2]
        #    if c[1][1]+c[1][3] > min_box[3]:
        #        min_box[3] = c[1][1]+c[1][3]

    #covered_surface = (min_box[2]-min_box[0]) * (min_box[3] - min_box[1])
    check_surface = (covered_surface/total_surface) > (1/4)

    return check_surface
  
def write_json(new_data, filename='data2.json'):
    """write in json file

    Args:
        new_data (dict): data to be written
        filename (str, optional): name of the file to write the data. Defaults to 'data2.json'.
    """
    if not os.path.exists(filename):
        json_object = json.dumps(new_data, indent = 4)
        with open(filename, "w") as outfile:
            outfile.write(json_object)
    #with open(filename,'r+') as file:
		# First we load existing data into a dict.
        #file_data = json.load(file)
		# Join new_data with file_data inside emp_details
        #file_data.append(new_data)
		# Sets file's current position at offset.
        #file.seek(0)
		# convert back to json.
        #json.dump(file_data, file, indent = 4)

def compute_obj_maps(df_comp):
    df_comp['index_col'] = df_comp.index
    obj_maps = df_comp.set_index('name').to_dict()['index_col']
    #print(obj_maps)
    return obj_maps

def random_augmentation(obj_maps,scene_graph,objects):
    relationships = []
    l1 = [i for i in range(len(obj_maps))]
    d1 = {key: [] for key in l1}
    choices = [True,False]
    for s in range(len(scene_graph) // 3):
        rel = scene_graph[(s*3)+1]
        id1 = obj_maps[scene_graph[s*3]["class"]]
        id2 = obj_maps[scene_graph[s*3+2]["class"]]
        if rel == "inside":
            #id1 in id2 d1 keys are parent id, so id2 is key
            objects[id1]["parent"] = id2
            d1[id2].append(id1)
        else:
            if(random.choice(choices)):
                relationships.append({"sub_id": id2, "predicate": opposite_relations[rel], "obj_id": id1})
            else:
                relationships.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
    for id2 in d1:
        l2 = d1[id2]
        if(len(l2) != 0):
            id1 = random.choice(l2)
            relationships.append({"sub_id": id1, "predicate": "inside", "obj_id": id2})
    return relationships,objects

def determinstic_augmentation(obj_maps,scene_graph,objects,size = 4):
    """
    determinstically select all components for all inside realtions the index add to the #of iterations
    Args:
        obj_maps: object mapping provided
        scene_graph: scene graph to augment
        objects: objects to add parent for inside relations
        size: number of relations we would like it to return default: 4 and objects
    Returns: tuple(list,list)
    """
    relationships = []
    l1 = [i for i in range(len(obj_maps))]
    d1 = {key: [] for key in l1}
    for s in range(len(scene_graph) // 3):
        rel = scene_graph[(s * 3) + 1]
        id1 = obj_maps[scene_graph[s * 3]["class"]]
        id2 = obj_maps[scene_graph[s * 3 + 2]["class"]]
        if rel == "inside":
            #id1 in id2 d1 keys are parent id, so id2 is key. id2 is parent id1 is child
            objects[id1]["parent"] = id2
            d1[id2].append(id1)
    d1 = {k: v for k, v in d1.items() if v} #remove all empty list from d1
    keys = d1.keys()
    for si in range(size):
        index = si
        iteration_iseven = (si%2 == 0)
        relationship = []
        choices = {} #choices for each relationship
        for key in keys:
            list_of_child = d1[key] #return a list of child
            list_of_child_is_even = (len(list_of_child)%2==0)
            if (iteration_iseven == list_of_child_is_even):
                choice = min(index, len(list_of_child) - 1) #index of the list_of_child we chose
            else:
                choice = min(index, len(list_of_child) - 2)
            choices[key] = list_of_child[choice] #store parent and its child in choices
            relationship.append({"sub_id": list_of_child[choice], "predicate": "inside", "obj_id": key})
            index -= choice
        children = choices.values()#grab all the children
        for s in range(len(scene_graph) // 3):
            rel = scene_graph[(s * 3) + 1]
            id1 = obj_maps[scene_graph[s * 3]["class"]]
            id2 = obj_maps[scene_graph[s * 3 + 2]["class"]]
            if rel != "inside":
                if(id1 in children):
                    relationship.append({"sub_id": id2, "predicate": opposite_relations[rel], "obj_id": id1})
                else:
                    relationship.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
        relationships.append(relationship)
    return relationships,objects

def random_skip_layer_augmentation(obj_maps,scene_graph,objects):
    relationships = []
    l1 = [i for i in range(len(obj_maps))]
    d1 = {key: [] for key in l1}
    choices = [True, False]
    todelete = [] #contain id that were deleted
    sibiling_rel = ["right","left","below","above"]
    for s in range(len(scene_graph) // 3):
        rel = scene_graph[(s * 3) + 1]
        id1 = obj_maps[scene_graph[s * 3]["class"]]
        # first find the class of the objects then use object maps to compute its corresponding integer
        id2 = obj_maps[scene_graph[s * 3 + 2]["class"]]
        # first find the class of the objects then use object maps to compute its corresponding integer
        if rel == "inside":
            #flag value ranges from 0 to 2. 2 being preserving the current structure and keeping the relationship of id1 in id2
            #1 being making id1 and id2 siblings. 0 being deleting id1 as a whole.
            flag = np.random.binomial(2, 0.9, 1)[0]
            if(flag == 2):
                # id1 in id2 d1 keys are parent id, so id2 is key
                print("not skipping")
                if(id2 in todelete):
                    id2_parent = objects[id2]["parent"]
                    objects[id1]["parent"] = id2_parent
                    d1[id2_parent].append(id1)
                else:
                    objects[id1]["parent"] = id2
                    d1[id2].append(id1)
            elif(flag == 0):
                if(id2 in todelete):
                    objects[id1]["parent"] = objects[id2]["parent"]
                    continue
                print("delete", id1)
                objects[id1]["parent"] = id2
                todelete.append(id1)
            else:
                if(id2 in todelete):
                    continue
                print("switch", id1 ,", " ,id2)
                id2_parent = objects[id2]["parent"]
                objects[id1]["parent"] = id2_parent
                d1[id2_parent].append(id1)
                new_rel = random.choice(sibiling_rel)
                relationships.append({"sub_id": id1, "predicate": new_rel, "obj_id": id2})
    for s in range(len(scene_graph) // 3):
        rel = scene_graph[(s * 3) + 1]
        id1 = obj_maps[scene_graph[s * 3]["class"]]
        # first find the class of the objects then use object maps to compute its corresponding integer
        id2 = obj_maps[scene_graph[s * 3 + 2]["class"]]
        # first find the class of the objects then use object maps to compute its corresponding integer
        if rel != "inside":
            if(id1 in todelete or id2 in todelete):
                continue
            elif (random.choice(choices)):
                relationships.append({"sub_id": id2, "predicate": opposite_relations[rel], "obj_id": id1})
            else:
                relationships.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
    for id2 in d1:
        l2 = d1[id2]
        if (len(l2) != 0):
            id1 = random.choice(l2)
            relationships.append({"sub_id": id1, "predicate": "inside", "obj_id": id2})
    counter = 0
    for de in todelete:
        del objects[de - counter]
        counter += 1
    return relationships, objects

def main():
    directory = '../data/screenshots/combined'
    df = pd.read_csv('../data/clay/clay_labels.csv')
    run_old = False
    with open('../data/clay/split_train_id.txt') as f:
        temp = f.readlines()
        train_ids = [int(x) for x in temp]
    with open('../data/clay/split_dev_id.txt') as f:
        temp = f.readlines()
        dev_ids = [int(x) for x in temp]
    with open('../data/clay/split_test_id.txt') as f:
        temp = f.readlines()
        test_ids = [int(x) for x in temp]
	#image1 = cv2.imread("./data/screenshots/combined/4.jpg")
	#image_gui = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	#image_gui = cv2.resize(image_gui, (1440, 2560))
	#plt.imshow(image_gui)
	
    train_list = []
    test_list = []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        name, extension = os.path.splitext(path)
		# checking if it is a file
        if os.path.isfile(path) and extension==".json":
            f = open (path, "r", encoding="utf-8")
            data = json.loads(f.read())
            f.close()
            components = [("ROOT", [0,0,1440,2560], None, 0)]
            objects = [{"box": [0,0,1440,2560], "class": "ROOT", "parent": 0}]
            items_dico = dict()
            if run_old:
                components, items_dico, objects = parse_json_old(data = data["activity"]["root"], df = df.loc[df['screen_id'] == int(name.split("/")[-1])],components=components, items_dico=items_dico, objects=objects)
            else:
                components, items_dico, objects = parse_json(data = data["activity"]["root"], df = df.loc[df['screen_id'] == int(name.split("/")[-1])],components=components, items_dico=items_dico, objects=objects)
            
            if check_coverage(components):
                
                df_comp = pd.DataFrame(components, columns=["name", "box", "parent", "deep"])
                
                scene_graph = []
                scene_graph2 = []
                unique = []
                for j in df_comp.parent.values.tolist():
                    if j not in unique:
                        unique.append(j)
                
                for i in unique:
                    if i == None:
                        i = "None"
                    filter = df_comp[df_comp['parent'] == i]
                    list_filter = filter.values.tolist()
                    sample = random.sample(list_filter, len(list_filter))
                    sample2 = random.sample(list_filter, len(list_filter))
                    for s in range(len(sample)-1):
                        scene_graph.append({"box": sample[s][1], "class": sample[s][0]})
                        relation = get_relation(sample[s][1], sample[s+1][1])
                        scene_graph.append(relation)
                        scene_graph.append({"box": sample[s+1][1], "class": sample[s+1][0]})
                        
                        scene_graph2.append({"box": sample2[s][1], "class": sample2[s][0]})
                        relation2 = get_relation(sample2[s][1], sample2[s+1][1])
                        scene_graph2.append(relation2)
                        scene_graph2.append({"box": sample2[s+1][1], "class": sample2[s+1][0]})
                    
                    if i != 'None':
                        for s in range(len(sample)):
                            scene_graph.append({"box": sample[s][1], "class": sample[s][0]})
                            scene_graph.append("inside")
                            scene_graph.append({"box": list(list(df_comp[df_comp['name'] == i]["box"].values)[0]), "class": i})
                            
                            scene_graph2.append({"box": sample2[s][1], "class": sample2[s][0]})
                            scene_graph2.append("inside")
                            scene_graph2.append({"box": list(list(df_comp[df_comp['name'] == i]["box"].values)[0]), "class": i})
                relationships = []
                objects_map = {}
                exist_inside = []
                for s in range(len(scene_graph)//3):
                    id1 = None
                    rel = None
                    id2 = None
                    if scene_graph[s*3]["class"] in objects_map:
                        id1 = objects_map[scene_graph[s*3]["class"]]
                    else:
                        index = 0
                        for o in objects:
                        #if scene_graph[s*3] == "Card_0":
                            #print(o, " ", scene_graph[s*3].split("_")[0], " ", used, " ", index not in used)
                            if o["class"] == scene_graph[s*3]["class"].split("_")[0] and o["box"] == scene_graph[s*3]["box"]:
                                objects_map[scene_graph[s*3]["class"]] = index
                                id1 = index
                                break
                            index += 1
                    if scene_graph[(s*3)+2]["class"] in objects_map:
                        id2 = objects_map[scene_graph[(s*3)+2]["class"]]
                    else:
                        index = 0
                        for o in objects:
                            if o["class"] == scene_graph[(s*3)+2]["class"].split("_")[0] and o["box"] == scene_graph[(s*3)+2]["box"]:
                                objects_map[scene_graph[(s*3)+2]["class"]] = index
                                id2 = index
                                break
                            index += 1
                    rel = scene_graph[(s*3)+1]
                    #if id1 == None:
                        #print(scene_graph[s*3], " ", objects_map)

                    if rel!="inside":
                        relationships.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
                    else:
                        objects[id1]["parent"] = id2

                        if id2 not in exist_inside:
                            relationships.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
                            exist_inside.append(id2)
                            
                            
                relationships2 = []
                objects_map2 = {}
                exist_inside2 = []
                for s in range(len(scene_graph2)//3):
                    id1 = None
                    rel = None
                    id2 = None
                    if scene_graph2[s*3]["class"] in objects_map2:
                        id1 = objects_map2[scene_graph2[s*3]["class"]]
                    else:
                        index = 0
                        for o in objects:
                        #if scene_graph[s*3] == "Card_0":
                            #print(o, " ", scene_graph[s*3].split("_")[0], " ", used, " ", index not in used)
                            if o["class"] == scene_graph2[s*3]["class"].split("_")[0] and o["box"] == scene_graph2[s*3]["box"]:
                                objects_map2[scene_graph2[s*3]["class"]] = index
                                id1 = index
                                break
                            index += 1
                    if scene_graph2[(s*3)+2]["class"] in objects_map2:
                        id2 = objects_map2[scene_graph2[(s*3)+2]["class"]]
                    else:
                        index = 0
                        for o in objects:
                            if o["class"] == scene_graph2[(s*3)+2]["class"].split("_")[0] and o["box"] == scene_graph2[(s*3)+2]["box"]:
                                objects_map2[scene_graph2[(s*3)+2]["class"]] = index
                                id2 = index
                                break
                            index += 1
                    rel = scene_graph2[(s*3)+1]
                    #if id1 == None:
                        #print(scene_graph[s*3], " ", objects_map)

                    if rel!="inside":
                        relationships2.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
                    else:
                        objects[id1]["parent"] = id2

                        if id2 not in exist_inside2:
                            relationships2.append({"sub_id": id1, "predicate": rel, "obj_id": id2})
                            exist_inside2.append(id2)
                
                types = []
                for i in range(len(objects)):
                    #objects[i]["box"] = bbox_conv(objects[i]["box"])
                    if objects[i]["class"] not in types:
                        types.append(objects[i]["class"])
                if len(types) > 2 and len(relationships)<=33:
                    dictionary ={
                        "objects" : objects,
                        "relationships" : relationships,
                        "width": 1440,
                        "height": 2560,
                        "id":int(name.split("/")[-1]),
                        "path":str(name.split("/")[-1]+".jpg")
                    }
                    if int(name.split("/")[-1]) in train_ids:
                        train_list.append(dictionary)
                    elif int(name.split("/")[-1]) in dev_ids:
                        train_list.append(dictionary)
                    elif int(name.split("/")[-1]) in test_ids:
                        test_list.append(dictionary)
                    
                    dictionary2 ={
                        "objects" : objects,
                        "relationships" : relationships2,
                        "width": 1440,
                        "height": 2560,
                        "id":int(name.split("/")[-1])+100000,
                        "path":str(str(int(name.split("/")[-1])+100000)+".jpg")
                    }
                    #if int(name.split("/")[-1]) in train_ids:
                    #    train_list.append(dictionary2)
                    #elif int(name.split("/")[-1]) in dev_ids:
                    #    train_list.append(dictionary2)
                    #elif int(name.split("/")[-1]) in test_ids:
                    #    test_list.append(dictionary2)

    print("train size: ", len(train_list))   
    print("test size: ", len(test_list))         
      
    write_json(train_list, filename="../data/clay/train.json")
    write_json(test_list, filename="../data/clay/test.json")


if __name__ == "__main__":
	main()

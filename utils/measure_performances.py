import numpy as np
import json
import os
import os.path
from scipy.stats import wasserstein_distance
import pandas as pd
import matplotlib.pyplot as plt

def conv_xywh2xyxyxy(bb):
    return bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]

def overlap_parent(data):
    total_loss = 0
    N = 0
    for screen in data:
        total_screen = 0
        for obj in screen["objects"]:
            if obj['class'] != "ROOT" and obj['class'] != None and obj['class'] != "[PAD]":
                bb1 = obj['box']
                #print(screen['id'], len(screen['objects']), obj['parent'])
                if screen['objects'][obj['parent']]['box'] == None:
                    parent = screen['objects'][0]['box']
                else:
                    parent = screen['objects'][obj['parent']]['box']
                if bb1[0] >= bb1[2] or bb1[1] >= bb1[3]:
                    print(screen['id'])
                assert bb1[0] < bb1[2]
                assert bb1[1] < bb1[3]
                assert parent[0] < parent[2]
                assert parent[1] < parent[3]

                # determine the coordinates of the intersection rectangle
                x_left = max(bb1[0], parent[0])
                y_top = max(bb1[1], parent[1])
                x_right = min(bb1[2], parent[2])
                y_bottom = min(bb1[3], parent[3])

                if x_right < x_left or y_bottom < y_top:
                    iou =  0.0
                else:

                    # The intersection of two axis-aligned bounding boxes is always an
                    # axis-aligned bounding box
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)

                    # compute the area of both AABBs
                    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
                    parent_area = (parent[2] - parent[0]) * (parent[3] - parent[1])

                    # compute the intersection over union by taking the intersection
                    # area and dividing it by the sum of prediction + ground-truth
                    # areas - the interesection area
                    iou = intersection_area / float(bb1_area)
                assert iou >= 0.0
                assert iou <= 1.0
                #if (1-iou) > 0.2:
                #    print(screen["id"], bb1, parent, (1-iou))
                total_loss += (iou)
                total_screen += (iou)
        #if total_screen > 17:
        #    print(screen["id"], " ", total_screen)
        #    print(screen['objects'])
                N += 1
    if N == 0:
        N = 1
    avg_loss = total_loss/N
    return (total_loss, avg_loss)



def overlap_components(data):
    total_loss = 0
    N = 0
    for screen in data:
        boxes_by_parent = dict()
        for obj in screen["objects"]:
            bb1 = obj['box']
            if obj['parent'] not in boxes_by_parent and obj["class"] != "ROOT" and obj['class'] != None and obj['class'] != "[PAD]":
                boxes_by_parent[obj['parent']] = [bb1]
            elif obj["class"] != "ROOT" and bb1 != None and obj['class'] != "[PAD]":
                for bb2 in boxes_by_parent[obj['parent']]:
                    assert bb1[0] < bb1[2]
                    assert bb1[1] < bb1[3]
                    assert bb2[0] < bb2[2]
                    assert bb2[1] < bb2[3]

                    # determine the coordinates of the intersection rectangle
                    x_left = max(bb1[0], bb2[0])
                    y_top = max(bb1[1], bb2[1])
                    x_right = min(bb1[2], bb2[2])
                    y_bottom = min(bb1[3], bb2[3])

                    if x_right < x_left or y_bottom < y_top:
                        iou = 0.0
                    else:

                        # The intersection of two axis-aligned bounding boxes is always an
                        # axis-aligned bounding box
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        # compute the area of both AABBs
                        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
                        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

                        # compute the intersection over union by taking the intersection
                        # area and dividing it by the sum of prediction + ground-truth
                        # areas - the interesection area
                        iou = intersection_area / min(float(bb1_area), float(bb2_area))
                    assert iou >= 0.0
                    assert iou <= 1.0
                    total_loss += (1-iou)
                    N += 1
                    
                boxes_by_parent[obj['parent']].append(bb1)
    if N == 0:
        N = 1
    avg_loss = total_loss/N
    return (total_loss, avg_loss)

def alignment(data):
    total_loss = 0
    N = 0
    for screen in data:
        for i in range(len(screen['objects'])):
            if screen["objects"][i]['box'] != None and screen["objects"][i]['class'] != '[PAD]':
                minimum_value = 999999
                for j in range(len(screen['objects'])):
                    minimum_edge = 0
                    if i != j:
                        obj1 = screen["objects"][i]['box']
                        obj2 = screen["objects"][j]['box']
                        if obj2 != None:
                            left = abs(obj1[0]/1440 - obj2[0]/1440)
                            middle = abs((obj1[0]/1440 + obj1[2]/1440)/2 - (obj2[0]/1440 + obj2[2]/1440)/2)
                            right = abs(obj1[2]/1440 - obj2[2]/1440)
                            minimum_edge = min(left, middle)
                            minimum_edge = min(minimum_edge, right)
                            minimum_value = min(minimum_value, minimum_edge)
                total_loss += minimum_value
                N += 1
    if N == 0:
        N = 1
    return (total_loss, total_loss/N)

def full_alignment(data):
    total_loss = 0
    N = 0
    for screen in data:
        for i in range(len(screen['objects'])):
            if screen["objects"][i]['box'] != None and screen["objects"][i]['class'] != '[PAD]':
                minimum_value = 999999
                for j in range(len(screen['objects'])):
                    minimum_edge = 0
                    if i != j and screen["objects"][j]['box'] != None and screen["objects"][j]['class'] != '[PAD]':
                        obj1 = screen["objects"][i]['box']
                        obj2 = screen["objects"][j]['box']
                        if obj2 != None:
                            left = abs(obj1[0]/1440 - obj2[0]/1440)
                            middle = abs((obj1[0]/1440 + obj1[2]/1440)/2 - (obj2[0]/1440 + obj2[2]/1440)/2)
                            right = abs(obj1[2]/1440 - obj2[2]/1440)
                            top = abs(obj1[1]/2560 - obj2[1]/2560)
                            middle_h = abs((obj1[1]/2560 + obj1[3]/2560)/2 - (obj2[1]/2560 + obj2[3]/2560)/2)
                            bottom = abs(obj1[3]/2560 - obj2[3]/2560)
                            minimum_edge = min(left, middle)
                            minimum_edge = min(minimum_edge, right)
                            minimum_edge = min(minimum_edge, top)
                            minimum_edge = min(minimum_edge, middle_h)
                            minimum_edge = min(minimum_edge, bottom)
                            minimum_value = min(minimum_value, minimum_edge)
                total_loss += minimum_value
                N += 1
    if N == 0:
        N = 1
    return (total_loss, (1-(total_loss/N)))
                    

def box_distribution(target_data, pred_data):
    target_x = np.zeros(1440)
    target_y = np.zeros(2560)
    target_w = np.zeros(1440)
    target_h = np.zeros(2560)
    for screen in target_data:
        for screen2 in pred_data:
            if screen['id'] == screen2['id']:
                for i in range(len(screen['objects'])):
                    if screen["objects"][i]['box'] != None:
                        index = int((screen["objects"][i]['box'][0]+screen["objects"][i]['box'][2])/2)
                        if index<0:
                            index = 0
                        elif index>=1440:
                            index = 1439
                        target_x[index] += 1
                        
                        index = int((screen["objects"][i]['box'][1]+screen["objects"][i]['box'][3])/2)
                        if index<0:
                            index = 0
                        elif index>=2560:
                            index = 2559
                        target_y[index] += 1
                        
                        index = int(screen["objects"][i]['box'][2] - screen["objects"][i]['box'][0])-1
                        if index<0:
                            index = 0
                        elif index>=1440:
                            index = 1439
                        target_w[index] += 1
                        
                        index = int(screen["objects"][i]['box'][3] - screen["objects"][i]['box'][1])-1
                        if index<0:
                            index = 0
                        elif index>=2560:
                            index = 2559
                        target_h[index] += 1
    target_x = target_x / np.linalg.norm(target_x)
    target_y = target_y / np.linalg.norm(target_y)
    target_w = target_w / np.linalg.norm(target_w)
    target_h = target_h / np.linalg.norm(target_h)
    
                
    pred_x = np.zeros(1440)
    pred_y = np.zeros(2560)
    pred_w = np.zeros(1440)
    pred_h = np.zeros(2560)
    for screen in pred_data:
        for i in range(len(screen['objects'])):
            if screen["objects"][i]['box'] != None and screen["objects"][i]['class'] != '[PAD]':
                index = int((screen["objects"][i]['box'][0]+screen["objects"][i]['box'][2])/2)
                if index<0:
                    index = 0
                elif index>=1440:
                    index = 1439
                pred_x[index] += 1
                
                index = int((screen["objects"][i]['box'][1]+screen["objects"][i]['box'][3])/2)
                if index<0:
                    index = 0
                elif index>=2560:
                    index = 2559
                pred_y[index] += 1
                
                index = int(screen["objects"][i]['box'][2] - screen["objects"][i]['box'][0])-1
                if index<0:
                    index = 0
                elif index>=1440:
                    index = 1439
                pred_w[index] += 1
                
                index = int(screen["objects"][i]['box'][3] - screen["objects"][i]['box'][1])-1
                if index<0:
                    index = 0
                elif index>=2560:
                    index = 2559
                pred_h[index] += 1
    #for i in range(len(target_x) - len(pred_x)):
    #    pred_x.append(0)
    #    pred_y.append(0)
    #    pred_w.append(0)
    #    pred_h.append(0)
    pred_x = pred_x / np.linalg.norm(pred_x)
    pred_y = pred_y / np.linalg.norm(pred_y)
    pred_w = pred_w / np.linalg.norm(pred_w)
    pred_h = pred_h / np.linalg.norm(pred_h)
    bins_x = list(range(1440))
    wd_x = wasserstein_distance(bins_x, bins_x, target_x, pred_x)
    bins_y = list(range(2560))
    wd_y = wasserstein_distance(bins_y, bins_y, target_y, pred_y)
    wd_w = wasserstein_distance(bins_x, bins_x, target_w, pred_w)
    wd_h = wasserstein_distance(bins_y, bins_y, target_h, pred_h)
    
    return wd_x, wd_y, wd_w, wd_h, 1-(wd_x + wd_y + wd_w + wd_h)/4000


def min_zero_row(zero_mat, mark_zero):
	
	'''
	The function can be splitted into two steps:
	#1 The function is used to find the row which containing the fewest 0.
	#2 Select the zero number on the row, and then marked the element corresponding row and column as False
	'''

	#Find the row
	min_row = [99999, -1]

	for row_num in range(zero_mat.shape[0]): 
		if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
			min_row = [np.sum(zero_mat[row_num] == True), row_num]

	# Marked the specific row and column as False
	zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
	mark_zero.append((min_row[1], zero_index))
	zero_mat[min_row[1], :] = False
	zero_mat[:, zero_index] = False

def mark_matrix(mat):

	'''
	Finding the returning possible solutions for LAP problem.
	'''

	#Transform the matrix to boolean matrix(0 = True, others = False)
	cur_mat = mat
	zero_bool_mat = (cur_mat == 0)
	zero_bool_mat_copy = zero_bool_mat.copy()

	#Recording possible answer positions by marked_zero
	marked_zero = []
	while (True in zero_bool_mat_copy):
		min_zero_row(zero_bool_mat_copy, marked_zero)
	
	#Recording the row and column positions seperately.
	marked_zero_row = []
	marked_zero_col = []
	for i in range(len(marked_zero)):
		marked_zero_row.append(marked_zero[i][0])
		marked_zero_col.append(marked_zero[i][1])

	#Step 2-2-1
	non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
	
	marked_cols = []
	check_switch = True
	while check_switch:
		check_switch = False
		for i in range(len(non_marked_row)):
			row_array = zero_bool_mat[non_marked_row[i], :]
			for j in range(row_array.shape[0]):
				#Step 2-2-2
				if row_array[j] == True and j not in marked_cols:
					#Step 2-2-3
					marked_cols.append(j)
					check_switch = True

		for row_num, col_num in marked_zero:
			#Step 2-2-4
			if row_num not in non_marked_row and col_num in marked_cols:
				#Step 2-2-5
				non_marked_row.append(row_num)
				check_switch = True
	#Step 2-2-6
	marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

	return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
	cur_mat = mat
	non_zero_element = []

	#Step 4-1
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					non_zero_element.append(cur_mat[row][i])
	min_num = min(non_zero_element)

	#Step 4-2
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					cur_mat[row, i] = cur_mat[row, i] - min_num
	#Step 4-3
	for row in range(len(cover_rows)):  
		for col in range(len(cover_cols)):
			cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
	return cur_mat

def hungarian_algorithm(mat): 
	dim = mat.shape[0]
	cur_mat = mat

	#Step 1 - Every column and every row subtract its internal minimum
	for row_num in range(mat.shape[0]): 
		cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])
	
	for col_num in range(mat.shape[1]): 
		cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])
	zero_count = 0
	while zero_count < dim:
		#Step 2 & 3
		ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
		zero_count = len(marked_rows) + len(marked_cols)

		if zero_count < dim:
			cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

	return ans_pos

def ans_calculation(mat, pos):
	total = 0
	ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
	for i in range(len(pos)):
		total += mat[pos[i][0], pos[i][1]]
		ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
	return total, ans_mat


def distance(bb1, bb2):
    return ((bb2[1] - bb1[1])**2 + (bb2[0] - bb1[0])**2)**0.5


            
    

def unique_matches(target_boxes, pred_boxes):
    used = []
    match = dict()
    score_array = []
    for ib1 in range(len(pred_boxes)):
        min_dist = 999999
        min_id = -1
        temp_array = []
        for ib2 in range(len(target_boxes)):
            bb1 = pred_boxes[ib1]['box']
            bb2 = target_boxes[ib2]['box']
            temp_array.append((min(bb1[2]*bb1[3], bb2[2]*bb2[3])**0.5) * (2**(-distance(bb1, bb2) - 2 * (abs(bb1[2] - bb2[2]) + abs(bb1[3] - bb2[3])))))
            if ib2 not in used and target_boxes[ib2]['class'] == pred_boxes[ib1]['class']:
                if distance(pred_boxes[ib1]['box'], target_boxes[ib2]['box']) < min_dist:
                    min_dist = distance(pred_boxes[ib1]['box'], target_boxes[ib2]['box'])
                    min_id = ib2
        used.append(min_id)
        match[ib1] = min_id
        score_array.append(temp_array)
    total = 0
    for key in match:
        if match[key] != -1:
            total += score_array[key][match[key]]
            
    fill_array = np.zeros(len(target_boxes))
            
    for i in range(len(target_boxes) - len(pred_boxes)):
        score_array.append(list(fill_array))
    
    np_score = np.array(score_array)
    max_value = np.max(np_score)
    cost_matrix = max_value - np_score
    ans_pos = hungarian_algorithm(cost_matrix.copy())#Get the element position.
    ans, ans_mat = ans_calculation(np_score, ans_pos)#Get the minimum or maximum value and corresponding matrix.µ
    res = total
    if ans != 0:
        res = res/ans
    return res

def unique_matches_find(target_data, pred_data):
    total_sim = 0
    with_threshold = 0
    N = 0
    for pred_screen in pred_data:
        pred_boxes = []
        target_boxes = []
        for o in pred_screen['objects']:
            if o['class'] != None and o['class'] != '[PAD]':
                pred_boxes.append(o)
            #else:
                #pred_boxes.append({'class': 'UNDEFINED', 'box': [0, 0, 0, 0], 'parent':None})
        found = False
        for target_screen in target_data:
            if int(target_screen['id']) == int(pred_screen['id']):
                for o in target_screen['objects']:
                    target_boxes.append(o)
                found = True
                break
        if found:
            sim = unique_matches(target_boxes, pred_boxes)
            total_sim += sim
            if sim>0:
                with_threshold += 1
            N+=1
    if N == 0:
        N = 1
    return total_sim, total_sim/N, with_threshold

def scene_graphs_matching(data, pred):
    total_loss = 0
    N = 0
    for pred_screen in pred:
        for target_screen in data:
            if pred_screen['id'] == target_screen['id']:
                for rel in range(len(pred_screen['relationships'])):
                    if pred_screen['relationships'][rel]['sub_id'] != -1 and pred_screen['relationships'][rel]['predicate'] != target_screen['relationships'][rel]['predicate']:
                        total_loss += 1
                    N += 1
    if N == 0:
        N = 1
    return total_loss, 1-total_loss/N


def get_inference_json(data, jsons_path, type_box):
    output = []
    for screen in data:
        if os.path.exists(os.path.join(jsons_path,(str(screen['id'])+type_box+".json"))):
        #if os.path.exists(os.path.join(jsons_path,("sg2im_"+str(screen['id'])+".json"))):
            f = open(os.path.join(jsons_path,(str(screen['id'])+type_box+".json")))
            #f = open(os.path.join(jsons_path,("sg2im_"+str(screen['id'])+".json")))
            temp = json.load(f)
            f.close()
            for i in range(len(temp['objects'])):
                if temp['objects'][i]['box'] != None:
                    temp['objects'][i]['box'][0] = temp['objects'][i]['box'][0]*1440/640
                    temp['objects'][i]['box'][1] = temp['objects'][i]['box'][1]*2560/640
                    temp['objects'][i]['box'][2] = temp['objects'][i]['box'][2]*1440/640
                    temp['objects'][i]['box'][3] = temp['objects'][i]['box'][3]*2560/640
                    if temp['objects'][i]['box'][2] <= temp['objects'][i]['box'][0]:
                        temp['objects'][i]['box'][2] = temp['objects'][i]['box'][0]+1
                    if temp['objects'][i]['box'][3] <= temp['objects'][i]['box'][1]:
                        temp['objects'][i]['box'][3] = temp['objects'][i]['box'][1]+1
                    if temp['objects'][i]['parent'] >= len(temp['objects']):
                        temp['objects'][i]['parent'] = 0
            output.append(temp)
    return output


def get_center(box):
    """return center of the bounding box

    Args:
        box (list(int)): bounding box

    Returns:
        tuple(int,int): coordinates of the center
    """
    width = box[2]-box[0]
    height = box[3] - box[1]
    x = box[0]+(width/2)
    y = box[1] + (height/2)
    return (x,y)


def get_new_relation(bb1, bb2):
    if bb1[0] > bb2[0] and bb1[1] > bb2[1] and bb1[2] < bb2[2] and bb1[3] < bb2[3]:
        return "inside"
    center1 = get_center(bb1)
    center2 = get_center(bb2)
    if abs(center1[0] - center2[0]) > abs(center1[1] - center2[1]):
        if center1[0] > center2[0]:
            return "right"
        else:
            return "left"
    else:
        if center1[1] > center2[1]:
            return "below"
        else:
            return "above"
        
def plot_metrics_by_cat(df, file_name):
    plt.cla()
    plt.clf()
    plt.close()
    
    #plt.plot(df['Category'], df['Overlap CP'], '-o', label='Overlap CP', color='red')
    #plt.plot(df['Category'], df['Overlap CC'], '-o', label='Overlap CC', color='blue')
    #plt.plot(df['Category'], df['Alignment'], '-o', label='Alignment', color='green')
    #plt.plot(df['Category'], df['W bbox'], '-o', label='W bbox', color='cyan')
    #plt.plot(df['Category'], df['DocSim'], '-o', label='DocSim', color='magenta')
    #plt.plot(df['Category'], df['SG error'], '-o', label='SG error', color='yellow')
    plt.bar(df['Category'], df['Overlap CP'], color='r')
    plt.bar(df['Category'], df['Overlap CC'], bottom=df['Overlap CP'], color='b')
    plt.bar(df['Category'], df['Alignment'], bottom=df['Overlap CP']+df['Overlap CC'], color='g')
    plt.bar(df['Category'], df['W bbox'], bottom=df['Overlap CP']+df['Overlap CC']+df['Alignment'], color='cyan')
    plt.bar(df['Category'], df['SG error'], bottom=df['Overlap CP']+df['Overlap CC']+df['Alignment']+df['W bbox'], color='magenta')
    plt.xlabel('Category')
    plt.legend(['CPI', 'CCS', 'Alignment', 'W bbox', 'GUI-AGC'],bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax = plt.gca()
    ax.set_xticklabels(labels=df['Category'],rotation=90)
    ax.figure.tight_layout()
    ax.figure.savefig(file_name)
    
def plot_metrics_by_type(df, file_name):
    plt.cla()
    plt.clf()
    plt.close()
    
    plt.plot(["3", "4", "5", "6", "7", "8", "9", "10", "11"], df['Overlap CP'], '-o', label='CPI', color='red')
    plt.plot(["3", "4", "5", "6", "7", "8", "9", "10", "11"], df['Overlap CC'], '-o', label='CCS', color='blue')
    plt.plot(["3", "4", "5", "6", "7", "8", "9", "10", "11"], df['Alignment'], '-o', label='Alignment', color='green')
    plt.plot(["3", "4", "5", "6", "7", "8", "9", "10", "11"], df['W bbox'], '-o', label='W bbox', color='cyan')
    #plt.plot(["3", "4", "5", "6", "7", "8", "9", "10", "11"], df['DocSim'], '-o', label='DocSim', color='magenta')
    plt.plot(["3", "4", "5", "6", "7", "8", "9", "10", "11"], df['SG error'], '-o', label='GUI-AGC', color='magenta')
    plt.xlabel('Number of unique component type')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax = plt.gca()
    ax.set_xticklabels(labels=["3", "4", "5", "6", "7", "8", "9", "10", "11"],rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(file_name)

def main():
    path_gt = "../data/clay/test.json"
    f = open(path_gt)
    data = json.load(f)
    f.close()
    
    isFalseInside = False
    
    if isFalseInside:
        true_parents_path = "../layoutTransformer/data/clay_v2/test.json"
        f = open(true_parents_path)
        data_correct = json.load(f)
        f.close()
        for screen in range(len(data)):
            for screen2 in data_correct:
                if data[screen]['id'] == screen2['id']:
                    for o in range(len(data[screen]['objects'])):
                        for o2 in range(len(screen2['objects'])):
                            if data[screen]['objects'][o]['box'][0] == screen2['objects'][o2]['box'][0] and data[screen]['objects'][o]['box'][1] == screen2['objects'][o2]['box'][1] and data[screen]['objects'][o]['box'][2] == screen2['objects'][o2]['box'][2] and data[screen]['objects'][o]['box'][3] == screen2['objects'][o2]['box'][3]:
                                corrParentId = 0
                                for correctParent in range(len(data[screen]['objects'])):
                                    if screen2['objects'][screen2['objects'][o2]['parent']]['box'][0] == data[screen]['objects'][correctParent]['box'][0] and screen2['objects'][screen2['objects'][o2]['parent']]['box'][1] == data[screen]['objects'][correctParent]['box'][1] and screen2['objects'][screen2['objects'][o2]['parent']]['box'][2] == data[screen]['objects'][correctParent]['box'][2] and screen2['objects'][screen2['objects'][o2]['parent']]['box'][3] == data[screen]['objects'][correctParent]['box'][3]:
                                        corrParentId = correctParent
                                        break
                                data[screen]['objects'][o]['parent'] = corrParentId
                    for r in range(len(data[screen]['relationships'])):
                        data[screen]['relationships'][r]['predicate'] = get_new_relation(data[screen]['objects'][data[screen]['relationships'][r]['sub_id']]['box'], data[screen]['objects'][data[screen]['relationships'][r]['obj_id']]['box'])
                    break
    
    tot_parent_overlap_loss, avg_parent_overlap_loss = overlap_parent(data)
    tot_components_overlap_loss, avg_components_overlap_loss = overlap_components(data)
    tot_alignment_loss, avg_alignment_loss = alignment(data)
    tot_full_alignment_loss, avg_full_alignment_loss = full_alignment(data)
    
    data_inference = get_inference_json(data, "../layoutTransformer/experiments/clay_seq2seq_bothOverlap/test_4/sg2im_json", "_r")
    #data_inference = get_inference_json(data, "../sg2im/output_jsonV2", "_r")
    
    
    tot_parent_overlap_loss_inference, avg_parent_overlap_loss_inference = overlap_parent(data_inference)
    tot_components_overlap_loss_inference, avg_components_overlap_loss_inference = overlap_components(data_inference)
    tot_alignment_loss_inference, avg_alignment_loss_inference = alignment(data_inference)
    tot_full_alignment_loss_inference, avg_full_alignment_loss_inference = full_alignment(data_inference)
    dist_x, dist_y, dist_w, dist_h, dist_tot = box_distribution(data, data_inference)
    total_unique_matches, avg_unique_matches, with_threshold = unique_matches_find(data, data_inference)
    total_sg, avg_sg = scene_graphs_matching(data, data_inference)
    
    measure_influence = True
    
    if measure_influence:
        app_details_path = "../data/clay/app_details.csv"
        ui_details_path = "../data/clay/ui_details.csv"
        
        app_details = pd.read_csv(app_details_path)
        ui_details = pd.read_csv(ui_details_path)
        merge = pd.merge(ui_details, app_details, on="App Package Name", how='left')[['UI Number','Category']]
        
        unique_cat = merge.Category.unique()
        
        data_cat = []
        data_cat_parent_overlap = []
        data_cat_components_overlap = []
        data_cat_alignment = []
        data_cat_wbox = []
        data_cat_docsim = []
        data_cat_sg_error = []
        
        for cat in unique_cat:
            if cat!="000 - 1":
                temp = merge[merge["Category"] == cat]
                json_temp = []
                for screen in data_inference:
                    if int(screen['id']) in temp['UI Number'].unique():
                        json_temp.append(screen)
                _, avg_parent_overlap_loss_inference_temp = overlap_parent(json_temp)
                _, avg_components_overlap_loss_inference_temp = overlap_components(json_temp)
                _, avg_full_alignment_loss_inference_temp = full_alignment(json_temp)
                if len(json_temp) > 0 :
                    _, _, _, _, dist_tot_temp = box_distribution(data, json_temp)
                else:
                    dist_tot_temp = 0
                _, avg_unique_matches_temp, _ = unique_matches_find(data, json_temp)
                _, avg_sg_temp = scene_graphs_matching(data, json_temp)
                
                data_cat.append(cat)
                data_cat_parent_overlap.append(avg_parent_overlap_loss_inference_temp)
                data_cat_components_overlap.append(avg_components_overlap_loss_inference_temp)
                data_cat_alignment.append(avg_full_alignment_loss_inference_temp)
                data_cat_wbox.append(dist_tot_temp)
                data_cat_docsim.append(avg_unique_matches_temp)
                data_cat_sg_error.append(avg_sg_temp)
        cat_influence_df = pd.DataFrame({"Category": data_cat, "Overlap CP": data_cat_parent_overlap, "Overlap CC": data_cat_components_overlap, "Alignment": data_cat_alignment, "W bbox": data_cat_wbox, "DocSim": data_cat_docsim, "SG error": data_cat_sg_error})
        plot_metrics_by_cat(cat_influence_df, "../layoutTransformer/experiments/clay_seq2seq_bothOverlap/performances_cat6.png")
        cat_influence_df.to_csv("../layoutTransformer/experiments/clay_seq2seq_bothOverlap/performances_cat6.csv")
        
        data_type = [[3], [4], [5], [6], [7], [8], [9], [10], [11]]
        data_type_parent_overlap = []
        data_type_components_overlap = []
        data_type_alignment = []
        data_type_wbox = []
        data_type_docsim = []
        data_type_sg_error = []
        for t in data_type:
            json_temp = []
            for screen in data_inference:
                unique_type = []
                for o in screen['objects']:
                    if o['class'] not in unique_type:
                        unique_type.append(o['class'])
                if len(unique_type) in t:
                    json_temp.append(screen)
            _, avg_parent_overlap_loss_inference_temp = overlap_parent(json_temp)
            _, avg_components_overlap_loss_inference_temp = overlap_components(json_temp)
            _, avg_full_alignment_loss_inference_temp = full_alignment(json_temp)
            if len(json_temp) > 0 :
                _, _, _, _, dist_tot_temp = box_distribution(data, json_temp)
            else:
                dist_tot_temp = 0
            _, avg_unique_matches_temp, _ = unique_matches_find(data, json_temp)
            _, avg_sg_temp = scene_graphs_matching(data, json_temp)
            
            data_type_parent_overlap.append(avg_parent_overlap_loss_inference_temp)
            data_type_components_overlap.append(avg_components_overlap_loss_inference_temp)
            data_type_alignment.append(avg_full_alignment_loss_inference_temp)
            data_type_wbox.append(dist_tot_temp)
            data_type_docsim.append(avg_unique_matches_temp)
            data_type_sg_error.append(avg_sg_temp)
        type_influence_df = pd.DataFrame({"Type": data_type, "Overlap CP": data_type_parent_overlap, "Overlap CC": data_type_components_overlap, "Alignment": data_type_alignment, "W bbox": data_type_wbox, "DocSim": data_type_docsim, "SG error": data_type_sg_error})
        plot_metrics_by_type(type_influence_df, "../layoutTransformer/experiments/clay_seq2seq_bothOverlap/performances_type6.png")
    
    with open('../layoutTransformer/experiments/clay_seq2seq_bothOverlap/performances_test44.txt', 'w') as f:
    #with open('../sg2im/performances_test4.txt', 'w') as f:
        f.write("|")
        f.write("Total overlap IoU with parents in Real data: ")
        f.write(str(tot_parent_overlap_loss))
        f.write("|")
        f.write("Average (per screen) overlap IoU with parents in Real data: ")
        f.write(str(avg_parent_overlap_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total overlap IoU between components sharing the same parent in Real data: ")
        f.write(str(tot_components_overlap_loss))
        f.write("|")
        f.write("Average (per screen) overlap IoU between components sharing the same parent in Real data: ")
        f.write(str(avg_components_overlap_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (only horizontal) between components in Real data: ")
        f.write(str(tot_alignment_loss))
        f.write("|")
        f.write("Average (per screen) alignment (only horizontal) between components in Real data: ")
        f.write(str(avg_alignment_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (horizontal and vertical) between components in Real data: ")
        f.write(str(tot_full_alignment_loss))
        f.write("|")
        f.write("Average (per screen) alignment (horizontal and vertical) between components in Real data: ")
        f.write(str(avg_full_alignment_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        
        f.write("|")
        f.write("Total overlap IoU with parents in Generated data: ")
        f.write(str(tot_parent_overlap_loss_inference))
        f.write("|")
        f.write("Average (per screen) overlap IoU with parents in Generated data: ")
        f.write(str(avg_parent_overlap_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total overlap IoU between components sharing the same parent in Generated data: ")
        f.write(str(tot_components_overlap_loss_inference))
        f.write("|")
        f.write("Average (per screen) overlap IoU between components sharing the same parent in Generated data: ")
        f.write(str(avg_components_overlap_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (horizontal) between components in Generated data: ")
        f.write(str(tot_alignment_loss_inference))
        f.write("|")
        f.write("Average (per screen) alignment (horizontal) between components in Generated data: ")
        f.write(str(avg_alignment_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (horizontal and vertical) between components in Generated data: ")
        f.write(str(tot_full_alignment_loss_inference))
        f.write("|")
        f.write("Average (per screen) alignment (horizontal and vertical) between components in Generated data: ")
        f.write(str(avg_full_alignment_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        f.write("|")
        f.write("Wasserstein distance in x distribution: ")
        f.write(str(dist_x))
        f.write("|")
        f.write("Wasserstein distance in y distribution: ")
        f.write(str(dist_y))
        f.write("|")
        f.write("Wasserstein distance in w distribution: ")
        f.write(str(dist_w))
        f.write("|")
        f.write("Wasserstein distance in h distribution: ")
        f.write(str(dist_h))
        f.write("|")
        f.write("Total Wasserstein distance: ")
        f.write(str(dist_tot))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        f.write("|")
        f.write("Total unique matches: ")
        f.write(str(total_unique_matches))
        f.write("|")
        f.write("Average unique matches: ")
        f.write(str(avg_unique_matches))
        f.write("|")
        f.write("Total unique matches with threshold: ")
        f.write(str(with_threshold))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        f.write("|")
        f.write("Total scene graph differences: ")
        f.write(str(total_sg))
        f.write("|")
        f.write("Average (per screen) scene_graph differences: ")
        f.write(str(avg_sg))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        
    """data_inference = get_inference_json(data, "../layoutTransformer/experiments/clay_seq2seq_baseline2/test_1/sg2im_json", "_c")
    
    
    tot_parent_overlap_loss_inference, avg_parent_overlap_loss_inference = overlap_parent(data_inference)
    tot_components_overlap_loss_inference, avg_components_overlap_loss_inference = overlap_components(data_inference)
    tot_alignment_loss_inference, avg_alignment_loss_inference = alignment(data_inference)
    tot_full_alignment_loss_inference, avg_full_alignment_loss_inference = full_alignment(data_inference)
    dist_x, dist_y, dist_w, dist_h, dist_tot = box_distribution(data, data_inference)
    total_unique_matches, avg_unique_matches = unique_matches_find(data, data_inference)
    total_sg, avg_sg = scene_graphs_matching(data, data_inference)
    
    with open('../layoutTransformer/experiments/clay_seq2seq_baseline2/performances_test1_coarse.txt', 'w') as f:
        f.write("|")
        f.write("Total overlap IoU with parents in Real data: ")
        f.write(str(tot_parent_overlap_loss))
        f.write("|")
        f.write("Average (per screen) overlap IoU with parents in Real data: ")
        f.write(str(avg_parent_overlap_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total overlap IoU between components sharing the same parent in Real data: ")
        f.write(str(tot_components_overlap_loss))
        f.write("|")
        f.write("Average (per screen) overlap IoU between components sharing the same parent in Real data: ")
        f.write(str(avg_components_overlap_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (horizontal) between components in Real data: ")
        f.write(str(tot_alignment_loss))
        f.write("|")
        f.write("Average (per screen) alignment (horizontal) between components in Real data: ")
        f.write(str(avg_alignment_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (horizontal and vertical) between components in Real data: ")
        f.write(str(tot_full_alignment_loss))
        f.write("|")
        f.write("Average (per screen) alignment (horizontal and vertical) between components in Real data: ")
        f.write(str(avg_full_alignment_loss))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        
        f.write("|")
        f.write("Total overlap IoU with parents in Generated data: ")
        f.write(str(tot_parent_overlap_loss_inference))
        f.write("|")
        f.write("Average (per screen) overlap IoU with parents in Real data: ")
        f.write(str(avg_parent_overlap_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total overlap IoU between components sharing the same parent in Generated data: ")
        f.write(str(tot_components_overlap_loss_inference))
        f.write("|")
        f.write("Average (per screen) overlap IoU between components sharing the same parent in Generated data: ")
        f.write(str(avg_components_overlap_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment between components in Generated data: ")
        f.write(str(tot_alignment_loss_inference))
        f.write("|")
        f.write("Average (per screen) alignment between components in Generated data: ")
        f.write(str(avg_alignment_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("|")
        f.write("Total alignment (horizontal and vertical) between components in Generated data: ")
        f.write(str(tot_full_alignment_loss_inference))
        f.write("|")
        f.write("Average (per screen) alignment (horizontal and vertical) between components in Generated data: ")
        f.write(str(avg_full_alignment_loss_inference))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        f.write("|")
        f.write("Wasserstein distance in x distribution: ")
        f.write(str(dist_x))
        f.write("|")
        f.write("Wasserstein distance in y distribution: ")
        f.write(str(dist_y))
        f.write("|")
        f.write("Wasserstein distance in w distribution: ")
        f.write(str(dist_w))
        f.write("|")
        f.write("Wasserstein distance in h distribution: ")
        f.write(str(dist_h))
        f.write("|")
        f.write("Total Wasserstein distance: ")
        f.write(str(dist_tot))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        f.write("|")
        f.write("Total unique matches: ")
        f.write(str(total_unique_matches))
        f.write("|")
        f.write("Average unique matches: ")
        f.write(str(avg_unique_matches))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")
        
        f.write("|")
        f.write("Total scene graph differences: ")
        f.write(str(total_sg))
        f.write("|")
        f.write("Average (per screen) scene_graph differences: ")
        f.write(str(avg_sg))
        f.write("|")
        f.write("\n")
        f.write("----------------------------------------------------------------------\n")"""

if __name__ == "__main__":
	main()

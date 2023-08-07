import multiprocessing
from joblib import Parallel, delayed
import sys
sys.path.append("../")
sys.path.append("./")
from model import Rel2Layout, RelEncoder, Rel2Bbox, Rel2RegLayout
from trainer import Trainer, PretrainTrainer, RegTrainer
from utils import ensure_dir
from loader.COCODataset import COCORelDataset
from loader.VGmsdnDataset import VGmsdnRelDataset
from loader.CLAYDataset import CLAYRelDataset
import torch.backends.cudnn as cudnn
from model import build_model
import argparse
import logging
import json
import yaml
import cv2
from bounding_box import bounding_box as bb
import os
import numpy as np
from utils import ensure_dir
import pickle
import torch
import random
from random import randrange
from tqdm import tqdm
import time

logger = logging.getLogger('inference')
num_cores = multiprocessing.cpu_count()


class Inference_CLAY():
    def __init__(self, save_dir, vocab_dict):
        self.save_dir = save_dir
        self.device = self._prepare_gpu()
        self.vocab_dict = vocab_dict
        self.auto_filter = 10
            
    def check_GT(self, idx, dataset):
        """
        input_token = COCO[0][0]
        input_obj_id = COCO[0][1]
        input_box_label = COCO[0][2]
        output_label = COCO[0][3]
        segment_label = COCO[0][4]
        token_type = COCO[0][5]
        """
        image_id = dataset.image_ids[idx]
        single_data = dataset[idx]
        log_file_name = os.path.join(self.save_dir,str(image_id)+'.txt')
        image_wh = dataset.image_id_to_size[image_id]
        image_size = [image_wh[1], image_wh[0], 3]
        
        box_mask = np.array(single_data[2]) != 2.
        boxes = np.array(single_data[2])[box_mask].reshape(-1,4)

        boxes = self.xywh2xyxy(boxes, image_wh)
        
        id_mask = np.array(single_data[1]) != 0.
        ids = np.array(single_data[1])[id_mask]
        
        class_mask = (single_data[0][1::2] != 0)
        clss = single_data[0][1::2][class_mask]
        
        __image__mask = clss != 4
        clss = clss[__image__mask]
        clss = self.idx2vocab(clss, 'text')

        boxes = boxes[__image__mask]
        
#         assert len(clss) == len(boxes) == len(ids), "{},{},{}".format(len(clss),len(boxes),len(ids))

        self.draw_img(image_size = image_size, boxes=boxes, labels=clss,
                      save_dir = self.save_dir,label_ids=[], name=str(image_id)+'_gt')
        
    def check_from_model(self, dataset_idx, dataset, model, random=False, layout_save=None):
        model.to(self.device)
        model.eval()
#         bb_gt = self.check_GT(idx, dataset)
        if random == True:
            dataset_idx = randrange(len(dataset.image_ids))
        image_id = dataset.image_ids[dataset_idx]
        single_data = dataset[dataset_idx]
        image_wh = dataset.image_id_to_size[image_id]
        if layout_save is not None: 
            ensure_dir(layout_save)
            image_wh = [640, 640]
        image_wh = [1280, 720]
        image_size = [image_wh[0], image_wh[1], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        json_save_dir = os.path.join(self.save_dir, 'sg2im_json')
        ensure_dir(json_save_dir)
        json_file_name_refine = os.path.join(json_save_dir, str(image_id)+'_r.json')
        json_file_name_coarse = os.path.join(json_save_dir, str(image_id)+'_c.json')

        input_token = single_data[0].unsqueeze(0).to(self.device)
        input_obj_id = single_data[1].unsqueeze(0).to(self.device)
        input_parent_id = single_data[3].unsqueeze(0).to(self.device)
        input_wh = single_data[5].unsqueeze(0).to(self.device)
        output_label = single_data[6].unsqueeze(0).to(self.device)
        segment_label = single_data[7].unsqueeze(0).to(self.device)
        token_type = single_data[8].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)
        global_mask = input_token >= 2
        #print("input token :", input_token)
        #print("input obj id: ", input_obj_id)
        #print("segment label: ", segment_label)
        #print("token type: ", token_type)
        #print("src mask: ", src_mask)

        
        with torch.no_grad():
            start = time.time()
            box_position_logits, size_position_logits, vocab_logits, obj_id_logits, parent_id_logits, token_type_logits, output_box, _, refine_box, _ = \
            model(input_token, input_obj_id, input_parent_id, segment_label, token_type, src_mask, trg_input_box=input_wh, inference=True, epoch=0, global_mask=global_mask)
            end = time.time()
        #print("Elapsed time,", end-start)
        #print("Batch Size", input_token.size(0))
        
        pred_vocab = vocab_logits.argmax(2)
        pred_id = obj_id_logits.argmax(2)
        #id_mask = (input_token == 3) * (pred_vocab > 4) * (pred_vocab < 155)
        #input_obj_id[id_mask] = pred_id[id_mask]
        
        # construct mask
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        #input_parent_id_list = list(input_parent_id[0].cpu().numpy())
        #mask = torch.zeros(1,len(input_obj_id_list))
        #for i in range(1, int(max(input_obj_id_list))+1):
        #    idx = input_obj_id_list.index(i)
        #    mask[0, idx] = 1
#             mask[0, input_obj_id_list.index(i)] = 1
        #mask = mask.bool()
        
#         print(pred_vocab)

#         print(output_box)
#         pred_mask = (pred_vocab >= 4) * (pred_vocab < 176)
        #print(mask)
        #print(mask.size(), pred_vocab.size())
        pred_vocab = pred_vocab.detach()
#         print(output_box[0, 2::4, :2])
        
        #output_box[0, 3::4, :2] = output_box[0, 1::4, :2] - 1*output_box[0, 2::4, :2]
        #refine_box[0, 3::4, :2] = refine_box[0, 1::4, :2] - 1*output_box[0, 2::4, :2]
        
        output_boxes = output_box.detach()
        refine_boxes = refine_box.detach()
        output_class_ids = input_obj_id.detach()
        #if self.auto_filter < len(output_boxes.squeeze(0)):
        #    return 0
        output_boxes = self.xcycwh2xyxy(output_boxes[0], image_wh)
        refine_boxes = self.xcycwh2xyxy(refine_boxes[0], image_wh)
        
        pred_classes = self.idx2vocab(pred_vocab[0], 'text')
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')
        input_sentence = self.idx2vocab(input_token.squeeze(0), 'text')

        #print(output_boxes.size())
        #print(output_boxes[0:10,:])
        #print(len(pred_classes))
        #print(len(output_sentence))
        #print(len(input_sentence))
        #print(output_class_ids[0])
        #print(input_wh[0,0:10,:])
        #print(output_boxes[0:10,:])
        #print(pred_classes)
        #print(output_class_ids)
        #print(image_id)
        #exit()
        #for i in reversed(range(len(output_boxes))):
        #    if pred_classes[i] == "ROOT":
        #        output_boxes = self.fit_root(output_boxes[i], output_boxes, image_wh)
        #        refine_boxes = self.fit_root(refine_boxes[i], refine_boxes, image_wh)
        #        break
        
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx)
        self.draw_img(image_size = image_size, boxes=refine_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx, mode='r')
        self.write_log(output_sentence, input_obj_id_list, log_file_name)
        self.write_json(input_sentence, input_obj_id_list, json_file_name_coarse, 
                        name=image_id, idx=dataset_idx, boxes=output_boxes.squeeze(0), parents_id=input_parent_id.squeeze(0))
        self.write_json(input_sentence, input_obj_id_list, json_file_name_refine, 
                        name=image_id, idx=dataset_idx, boxes=refine_boxes.squeeze(0), parents_id=input_parent_id.squeeze(0))
        if layout_save is not None: 
            self.save_layout(boxes=output_boxes.squeeze(0), objs=pred_classes, 
                         save_path=layout_save, label_ids=output_class_ids.squeeze(0),
                         name=image_id, image_wh=image_wh)


    def fit_root(self, root_box, boxes, image_wh):
        boxes[:,0] = boxes[:,0] - root_box[0]
        boxes[:,1] = boxes[:,1] - root_box[1]
        boxes[:,2] = (image_wh[0]/root_box[2]) * boxes[:,2]
        boxes[:,3] = (image_wh[1]/root_box[3]) * boxes[:,3]
        return boxes


    def save_relation_prior(self, rel_classes, rel_prior_xy, save_dir):
        rel_prior_xy = rel_prior_xy.tolist()
        try:
            with open(os.path.join(save_dir, 'rel_prior.json'), 'r') as fp:
                rel_dict = json.load(fp)
        except:
            rel_dict = dict()
        for i, rel in enumerate(rel_classes):
            if rel in rel_dict.keys():
                rel_dict[rel].append(rel_prior_xy[i])
            else:
                rel_dict[rel] = [rel_prior_xy[i]]
        with open(os.path.join(save_dir, 'rel_prior.json'), 'w') as fp:
            json.dump(rel_dict, fp)

    def check_from_sg(self, input_dict, model, layout_save=None):
        model.to(self.device)
        model.eval()
        
        print("self input")
        image_id = input_dict['image_id']
        dataset_idx = input_dict['dataset_idx']
        image_wh = [640, 640]
        
        if layout_save is not None: 
            ensure_dir(layout_save)
            image_wh = [640, 640]
        image_size = [image_wh[1], image_wh[0], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        json_save_dir = os.path.join(self.save_dir, 'sg2im_json')
        ensure_dir(json_save_dir)
        json_file_name = os.path.join(json_save_dir, str(image_id)+'.json')

        input_token = input_dict['tensor_list'][0].unsqueeze(0).to(self.device)
        input_obj_id = input_dict['tensor_list'][1].unsqueeze(0).to(self.device)
        segment_label = input_dict['tensor_list'][2].unsqueeze(0).to(self.device)
        token_type = input_dict['tensor_list'][3].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)
        global_mask = input_token >= 2
        
        with torch.no_grad():
            vocab_logits, obj_id_logits, parent_id_logits, token_type_logits, output_box, _, refine_box, _ = \
                model(input_token, input_obj_id, segment_label, token_type, src_mask, inference=True, epoch=0, global_mask=global_mask)


        pred_vocab = vocab_logits.argmax(2)
        pred_id = obj_id_logits.argmax(2)
        id_mask = (input_token == 3) * (pred_vocab > 4) * (pred_vocab < 155)
        input_obj_id[id_mask] = pred_id[id_mask]
        
        # construct mask
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        mask = torch.zeros(1,len(input_obj_id_list))
        for i in range(1, int(max(input_obj_id_list))+1):
            idx = len(input_obj_id_list) - 1 - input_obj_id_list[::-1].index(i)
            mask[0, idx] = 1

        mask = mask.bool()

        pred_vocab = pred_vocab[mask].detach()
        
        output_box[0, 3::4, :2] = output_box[0, 1::4, :2] - output_box[0, 2::4, :2]
    
        output_boxes = output_box[mask].detach()
        refine_boxes = refine_box[mask].detach()
        output_class_ids = input_obj_id[mask].detach()
        
        
        
        output_boxes = self.xcycwh2xyxy(output_boxes, image_wh)
        refine_boxes = self.xcycwh2xyxy(refine_boxes, image_wh)
        pred_classes = self.idx2vocab(pred_vocab, 'text')
        
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')

        input_sentence = self.idx2vocab(input_token.squeeze(0), 'text')
#         print(self.get_iou(output_boxes.squeeze(0), bb_gt))

        
        
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx)
        self.draw_img(image_size = image_size, boxes=refine_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx, mode='r')
        self.write_log(output_sentence, input_obj_id_list, log_file_name, 
                        name=image_id, idx=dataset_idx)
#         self.write_json(output_sentence, input_obj_id_list, json_file_name, 
#                         name=image_id, idx=dataset_idx)
        if layout_save is not None: 
            self.save_layout(boxes=output_boxes.squeeze(0), objs=pred_classes, 
                         save_path=layout_save, label_ids=output_class_ids.squeeze(0),
                         name=image_id, image_wh=image_wh)
            
    def draw_img(self, image_size, boxes, labels, label_ids, save_dir, name, idx, mode='c'):
        # setting color
        color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
                'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
        image = np.full(image_size, 200.)
        
        boxes[boxes < 0] = 1
        boxes[boxes > image_size[0]] = image_size[0] - 1
        drawn = []
        for i in reversed(range(len(boxes))):
            if label_ids[i] != 0 and label_ids[i] not in drawn:
                drawn.append(label_ids[i])
                bb.add(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                    str(labels[i]+'[{}]'.format(label_ids[i])), 
                    color=color[ord(labels[i][0])-ord('a')])
        self.show_and_save(image,
                   os.path.join(save_dir,str(name) + '_{}_{}_{}'.format(idx, mode,
                                         len(boxes))  +'.png'))
#         logger.info("Save image in {}".format(os.path.join(save_dir,str(name)+'_{}_{}'.format(idx, mode)+'.png')))
        
    def write_log(self, sentence, class_ids, log_file_name, name=None, idx=None):
        f = open(log_file_name, 'w')
        
        for i in range(1, len(sentence), 4):
            if sentence[i+1] == '[SEP]' and sentence[i+2] == '[SEP]':
                break
            single_pair = ''
            single_pair += sentence[i] + '[{}]'.format(class_ids[i]) + ' '
            single_pair += sentence[i+1] + ' '
            single_pair += sentence[i+2] + '[{}]'.format(class_ids[i+2]) + ' '
            single_pair += '\n'
            f.write(single_pair)
#         logger.info("Save log file in {}".format(log_file_name))


    def get_center(self, box):
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

    def get_relation(self, box1, box2):
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
        """center1 = self.get_center(box1)
        center2 = self.get_center(box2)
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
        """
        
    def write_json(self, sentence, class_ids, log_file_name, name=None, idx=None, boxes=None, parents_id=None):
        out_dict = dict()
        out_dict['id'] = name
        out_dict['dataset_idx'] = idx
        out_dict['objects'] = [{"box":None, "class": None, "parent":None} for i in range(max(class_ids))]
        out_dict['relationships'] = []
        for i in range(1, len(sentence), 4):
            if sentence[i] == '[PAD]':
                break
            out_dict['objects'][int(class_ids[i]-1)]["class"] = sentence[i]
            out_dict['objects'][int(class_ids[i]-1)]["box"] = boxes[i].tolist()
            out_dict['objects'][int(class_ids[i]-1)]["parent"] = parents_id[i].item()
            out_dict['objects'][int(class_ids[i+2]-1)]["class"] = sentence[i+2]
            out_dict['objects'][int(class_ids[i+2]-1)]["box"] = boxes[i+2].tolist()
            out_dict['objects'][int(class_ids[i+2]-1)]["parent"] = parents_id[i+2].item()
            single_rel = {"sub_id":int(class_ids[i]-1), "predicate":sentence[i+1], "obj_id":int(class_ids[i+2]-1)}
            out_dict['relationships'].append(single_rel)
        for j in range(len(out_dict['relationships'])):
            bb1 = out_dict['objects'][out_dict['relationships'][j]["sub_id"]]["box"]
            bb2 = out_dict['objects'][out_dict['relationships'][j]["obj_id"]]["box"]
            if bb1[0]>=bb2[0] and bb1[1]>=bb2[1] and bb1[2]<=bb2[2] and bb1[3]<=bb2[3]:
                out_dict['relationships'][j]["predicate"] = "inside"
            else:
                out_dict['relationships'][j]["predicate"] = self.get_relation(bb1, bb2)
        with open(log_file_name, 'w') as outfile:
            json.dump(out_dict, outfile)
#         logger.info("Save json file in {}".format(log_file_name))
        
    def save_layout(self, boxes, objs, save_path, label_ids, name, image_wh):
        output_dict = dict()
        output_dict['image_id'] = name
        output_dict['boxes'] = (boxes/image_wh[0]).tolist()
        output_dict['classes'] = objs
        output_dict['class_ids'] = label_ids.tolist()
        output_file_name = os.path.join(save_path,str(name)+'.json')
        with open(output_file_name, 'w') as fp:
            json.dump(output_dict, fp)
#         logger.info("Save json file in {}".format(output_file_name))
        return 0
    
    def xywh2xyxy(self, boxes, image_wh):
        center = boxes[:,:2].clone()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        return boxes
    
    def xcycwh2xyxy(self, boxes, image_wh):
        #center = boxes[:,:2].clone()
        #boxes[:,:2] = center - boxes[:,2:]/2.
        #boxes[:,2:] = center + boxes[:,2:]/2.
        boxes[:,0] = boxes[:,0] * image_wh[1] - boxes[:,2] *image_wh[1]/2
        boxes[:,1] = boxes[:,1] * image_wh[0] - boxes[:,3] *image_wh[0]/2
        boxes[:,2] = boxes[:,2] * image_wh[1] + boxes[:,0]
        boxes[:,3] = boxes[:,3] * image_wh[0] + boxes[:,1]

        #boxes[:,2] = boxes[:,2] - boxes[:,0]
        #boxes[:,3] = boxes[:,3] - boxes[:,1]
        
        
        return boxes

    def xywh_conv(self, boxes, image_wh):
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        return boxes
    
    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device
    
    def idx2vocab(self, idx, modality):
        sent = []
        for i in range(len(idx)):
            if modality == 'text':
                sent.append(self.vocab_dict[int(idx[i])])
            else:
                sent.append(self.cls_dict[idx[i]])
        return sent
    
    def show_and_save(self, image, path):
        cv2.imwrite(path, image)
        
    def run(self, cfg, model, dataset):
        seed_value = 5

        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        layout_path = None if cfg['TEST']['LAYOUT_MODE'] == "" else cfg['TEST']['LAYOUT_MODE']
        if cfg['TEST']['MODE'] == 'gt':
            if cfg['TEST']['RANDOM']: logger.warning('Test gt mode do not support random.')
            self.check_GT(cfg['TEST']['SAMPLE_IDX'], dataset=dataset)
        elif cfg['TEST']['MODE'] == 'model':
            if cfg['TEST']['RANDOM']: 
                if cfg['TEST']['SAMPLE_IDX'] == -1:
                    for idx in tqdm(range(3000)):
                        self.check_from_model(idx, dataset = dataset, model=model, 
                                              random=True, layout_save=layout_path)

                else:
                    self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset, model, 
                                       random=True, layout_save=layout_path)
            else: 
                if cfg['TEST']['SAMPLE_IDX'] == -1:
#                     processed_list = Parallel(n_jobs=num_cores)(delayed(self.check_from_model)(idx, idx, dataset = dataset, model=model,
#                           layout_save=layout_path) for idx in tqdm(range(3000)))

                    for idx in tqdm(range(3000)):
                        self.check_from_model(idx, dataset = dataset, model=model,
                                             layout_save=layout_path)
                else:
                    self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset=dataset,
                                         model=model, layout_save=layout_path)
        else:
            logger.error('We only support gt and model test mode.')
        
if __name__ == '__main__':
    data_dir = 'data/clay/'
    test_output = 'saved/clay_F_seq2seq_v11/test'
    cfg_path = './configs/clay/clay_seq2seq_v11.yaml'
#     int_json_path ='data/vg_msdn/stuff_train2017.json' 
    model_path = 'saved/clay_F_seq2seq_v11/checkpoint_4_0.18535177585640755.pth'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    vocab_dic_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
    with open(vocab_dic_path, 'rb') as file:
        vocab_dict = pickle.load(file)
        
    infer = Inference_CLAY(save_dir = test_output, vocab_dict = vocab_dict)
    
    ## dataset
    ins_data_path = os.path.join(data_dir, 'test.json')
    cat_path = os.path.join(data_dir, 'categories.json')
    dict_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
    
    dataset = CLAYRelDataset(instances_json_path = ins_data_path,
                                   category_json_path = cat_path, 
                                   dict_save_path = dict_path,
                                   sentence_size=128)
            
    # build model
    model = build_model(cfg)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    da = ['bush', 'kite', 'pant', 'laptop', 'paper', 'shoe', 'chair', 'ground', 'tire', 'cup', 'sky', 'bench', 'window', 'bike', 'board', 'orange', 'hat', 'hill', 'plate', 'woman', 'handle', 'animal', 'food', 'bear', 'wave', 'giraffe', 'background', 'desk', 'foot', 'shadow', 'lady', 'shelf', 'bag', 'sand', 'nose', 'rock', 'sidewalk', 'motorcycle', 'fence', 'people', 'house', 'sign', 'hair', 'street', 'zebra', 'mirror', 'racket', 'logo', 'girl', 'arm', 'flower', 'leaf', 'clock', 'dirt', 'boat', 'bird', 'umbrella', 'leg', 'bathroom', 'surfer', 'water', 'sink', 'trunk', 'post', 'tower', 'box', 'boy', 'cow', 'skateboard', 'roof', 'pillow', 'road', 'ski', 'wall', 'number', 'pole', 'table', 'cloud', 'sheep', 'horse', 'eye', 'top', 'neck', 'tail', 'vehicle', 'banana', 'fork', 'head', 'door', 'bus', 'glass', 'train', 'child', 'line', 'ear', 'reflection', 'car', 'tree', 'bed', 'cat', 'donut', 'cake', 'grass', 'toilet', 'player', 'airplane', 'ocean', 'glove', 'helmet', 'shirt', 'floor', 'bowl', 'snow', 'couch', 'field', 'lamp', 'book', 'branch', 'elephant', 'tile', 'beach', 'pizza', 'wheel', 'picture', 'plant', 'sandwich', 'mountain', 'track', 'hand', 'plane', 'stripe', 'letter', 'skier', 'vase', 'man', 'building', 'short', 'surfboard', 'phone', 'light', 'counter', 'dog', 'face', 'jacket', 'person', 'part', 'truck', 'bottle', 'jean', 'wing', '__in_image__', 'and', 'in_a', 'cover', 'over', 'at', 'have', 'in', 'carry', 'rid', 'have_a', 'inside_of', 'wear_a', 'for', 'in_front_of', 'hang_on', 'on_top_of', 'below', 'eat', 'beside', 'behind', 'above', 'under', 'on_front_of', 'lay_on', 'around', 'on_a', 'look_at', 'sit_on', 'between', 'watch', 'wear', 'walk_on', 'be_in', 'along', 'hold', 'with', 'by', 'stand_on', 'on', 'next_to', 'on_side_of', 'attach_to', 'of', 'inside', 'be_on', 'hang_from', 'near', 'sit_in', 'stand_in', 'of_a']
    
    # my own input
    sentence = ['[CLS]', 'tree', 'be_in', 'grass', '[SEP]',
                'tree', 'below', 'sky', '[SEP]',
                'man', 'on_top_of', 'tree', '[SEP]']
    sentence = ['[CLS]', 'tree', 'below', 'man', '[SEP]',
                         'tree', 'be_in', 'grass', '[SEP]']
#                 'man', 'wear_a', 'glass', '[SEP]']  
    for i in range(len(sentence)):
        sentence[i] = dataset.vocab['object_pred_name_to_idx'][sentence[i]]
        
    obj_id = [ 0, 2, 0, 3, 0,
                  2, 0, 1, 0,
                  4, 0, 2 ,0]
    obj_id = [0, 1, 0, 2, 0,
                 1, 0, 3, 0]
#                   5, 0, 6, 0]
    
    segment_label = [1]
    token_label = [0]
    for i in range(int((len(obj_id)-1)/4)):
        for j in range(4):
            segment_label.append(i+1)
            token_label.append(j+1)

    for i in range(128 - len(sentence)): 
        sentence.append(0)  
        obj_id.append(0)
        segment_label.append(0)
        token_label.append(0)

    self_input_dict = {'input_token':torch.LongTensor(sentence),
                       'input_obj_id':torch.LongTensor(obj_id),
                       'segment_label':torch.LongTensor(segment_label),
                      'token_type':torch.LongTensor(token_label),
                      'image_id':11116,
                      'image_wh':(800,600)}
    
    
    infer.check_from_model(100, dataset, model, self_input_dict = self_input_dict)
    

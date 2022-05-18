import torch
import cv2
import os
import time
import datetime
import math
import copy
import random
from torch.utils.data import Dataset, DataLoader
import sys
import json
from PIL import Image
import torchvision.transforms.functional as transform
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.utils.tensorboard import SummaryWriter
import numpy
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
from spellchecker import SpellChecker

# Calculates the Intersection Over Union for two specified bounding boxes
def calc_iou(bb1, bb2):
    # Get the coordinates of the intersecting box
    inter_x = max(bb1[0], bb2[0])
    inter_y = max(bb1[1], bb2[1])
    inter_x2 = min(bb1[2], bb2[2])
    inter_y2 = min(bb1[3], bb2[3])
    
    if inter_x2 < inter_x or inter_y2 < inter_y:
        return 0.0
    
    inter_area = (inter_x2 - inter_x) * (inter_y2 - inter_y)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = inter_area / float(bb1_area + bb2_area - inter_area)
    return iou

# Calculates the percentage of overlap between two bounding boxes determined by the first ones area 
def calc_overlap(bb1, bb2):
    # Get the coordinates of the intersecting box
    inter_x = max(bb1[0], bb2[0])
    inter_y = max(bb1[1], bb2[1])
    inter_x2 = min(bb1[2], bb2[2])
    inter_y2 = min(bb1[3], bb2[3])
    
    if inter_x2 < inter_x or inter_y2 < inter_y:
        return 0.0
    
    inter_area = (inter_x2 - inter_x) * (inter_y2 - inter_y)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    overlap = inter_area / float(bb1_area)
    return overlap

# Calculates the intersection area between two bounding boxes
def calc_intersection(bb1, bb2):
    inter_x = max(bb1[0], bb2[0])
    inter_y = max(bb1[1], bb2[1])
    inter_x2 = min(bb1[2], bb2[2])
    inter_y2 = min(bb1[3], bb2[3])
    
    if inter_x2 < inter_x or inter_y2 < inter_y:
        return 0.0
    
    return (inter_x2 - inter_x) * (inter_y2 - inter_y)

# Calculates the area of a bounding box
def calc_area(bb):
    return (bb[2] - bb[0]) * (bb[3] - bb[1])

# Padds a bounding box by a specific number, doubles the padding if text is specified
def pad_bb(bb, pad, text=False):
    x,y,x2,y2 = bb
    if text:
        return [x-pad*2, y-pad, x2+pad*2, y2+pad]
    return [x-pad, y-pad, x2+pad, y2+pad]

# Returns the smallest bounding box between two specified boxes
def return_smallest(bb1, bb2):
    bb1_x,bb1_y,bb1_x2,bb1_y2 = bb1
    bb2_x,bb2_y,bb2_x2,bb2_y2 = bb2
    bb1_size = (bb1_x2-bb1_x)*(bb1_y2-bb1_y)
    bb2_size = (bb2_x2-bb2_x)*(bb2_y2-bb2_y)
    
    return bb2 if bb1_size > bb2_size else bb1d

# Gets the bounding boxes from an image by processing the image
def get_bbs_from_image(im, clean=True, pad=0, text=False, ignore_padding=10, combine_all=False):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    bbs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        #cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 255), 2)
        if x > ignore_padding and y > ignore_padding and x < im.shape[:2][1]-ignore_padding and y < im.shape[:2][0]-ignore_padding:
            bbs.append([x,y,x+w,y+h])
                
    t_bbs = []
    [t_bbs.append(x) for x in bbs if x not in t_bbs]
    
    t_bbs = remove_small_bb_list(t_bbs, 10000)
    combined_bbs = combine_bb_list(t_bbs, pad=pad, text=text)
    
    if clean:
        combined_bbs = clean_bb_list(combined_bbs, pad=pad)
    
    if combine_all:
        temp_bb = combined_bbs[0]
        for bb in combined_bbs:
            if temp_bb[0] > bb[0]:
                temp_bb[0] = bb[0]
            if temp_bb[1] > bb[1]:
                temp_bb[1] = bb[1]
            if temp_bb[2] < bb[2]:
                temp_bb[2] = bb[2]
            if temp_bb[3] < bb[3]:
                temp_bb[3] = bb[3]
        return [temp_bb]
    return combined_bbs

# Iterates over a list of bunding boxes and combines intersecting bounding boxes into one
def combine_bb_list(bb_list, pad=0, text=False):
    bbs = bb_list.copy()
    iou_non_zero = True
    while iou_non_zero:
        iou_non_zero = False
        for i in range(len(bbs)-1):
            for c in range(i, len(bbs)):
                if bbs[i] == bbs[c]:
                    continue
                    
                iou = calc_iou(pad_bb(bbs[i], pad, text=text), bbs[c])
                
                if iou != 0:
                    iou_non_zero = True
                    bb = combine_bb(bbs[i], bbs[c])
                    bb1 = bbs[i].copy()
                    bb2 = bbs[c].copy()
                    
                    bbs.remove(bb1)
                    bbs.remove(bb2)
                    bbs.append(bb)
                    break;
            if iou_non_zero:
                break;
    return bbs

# Combines two specified bounding boxes into one
def combine_bb(bb1, bb2):
    bb1_x,bb1_y,bb1_x2,bb1_y2 = bb1
    bb2_x,bb2_y,bb2_x2,bb2_y2 = bb2

    if bb2_x < bb1_x:
        bb1_x = bb2_x
    if bb2_y < bb1_y:
        bb1_y = bb2_y
    if bb2_x2 > bb1_x2:
        bb1_x2 = bb2_x2
    if bb2_y2 > bb1_y2:
        bb1_y2 = bb2_y2
        
    return [bb1_x, bb1_y, bb1_x2, bb1_y2]

# Removes the smallest bounding box between bounding boxes which intersects
def clean_bb_list(bb_list, pad=0, text=False):
    bbs = bb_list.copy()
    iou_non_zero = True
    while iou_non_zero:
        iou_non_zero = False
        for i in range(len(bbs)):
            if i == len(bbs)-1:
                break;
                
            iou = calc_iou(pad_bb(bbs[i], pad, text=text), bbs[i+1])

            if iou == 0:
                continue

            iou_non_zero = True
            bb = return_smallest(bbs[i], bbs[i+1])
            bbs.remove(bb)
            break;
                
    return bbs

# Removes all bounding boxes which has a lower area than the pre specified parameter
def remove_small_bb_list(bb_list, size):
    cleaned_list = []
    for bb in bb_list:
        x,y,x2,y2 = bb
        w = x2-x
        h = y2-y
        if w*h > size:
            cleaned_list.append(bb)
            
    return cleaned_list

# Normalizes a pixel specific bounding box [x, y, x2, y2] to normalized bounding box [x, y, w, h]
def normalize_bb(bb, shape):
    h_img,w_img = shape
    x,y,x2,y2 = bb
    norm_w,norm_h = [(x2-x)/w_img, (y2-y)/h_img]
    return [((x+x2)/2)/w_img, ((y+y2)/2)/h_img, norm_w, norm_h]

# Denormalizes a normalized bounding box [x, y, w, h] to pixel specific bounding box [x, y, x2, y2]
def denormalize_bb(bb, shape):
    h_img,w_img = shape
    x,y,w,h = bb
    x_min,y_min = [int(x*w_img-(w*w_img)/2), int(y*h_img-(h*h_img)/2)]
    return [x_min, y_min, x_min+int(w*w_img), y_min+int(h*h_img)]

# Stringifies a bounding box for output
def bb_to_str(bb):
    return str(bb[0])+' '+str(bb[1])+' '+str(bb[2])+' '+str(bb[3])

# Destringifies a bounding box
def str_to_bb(bb_str):
    str_arr = bb_str.split(' ')
    return [float(str_arr[0]), float(str_arr[1]), float(str_arr[2]), float(str_arr[3]), float(str_arr[4])]

# Generates dataset structure by generating boundingbox labels, spliting data into train and validition sets
# also providing the found boundingboxes for verification of labeling being successfull 
def generate_dataset(root_folder, labels=[], split_components=True, train_val_ratio=0.8, combine_all=False):
    os.mkdir('./'+root_folder+'_generated/')
    os.mkdir('./'+root_folder+'_generated/images/')
    if train_val_ratio != 1:
        os.mkdir('./'+root_folder+'_generated/images/train/')
        os.mkdir('./'+root_folder+'_generated/images/val/')
    os.mkdir('./'+root_folder+'_generated/images/bbs/')
    os.mkdir('./'+root_folder+'_generated/labels/')

    if (split_components):    
        for component in os.listdir('./'+root_folder):
            if train_val_ratio != 1:
                os.mkdir('./'+root_folder+'_generated/images/train/'+component+'/')
                os.mkdir('./'+root_folder+'_generated/images/val/'+component+'/')
            else:
                os.mkdir('./'+root_folder+'_generated/images/'+component+'/')
			
            images = os.listdir('./'+root_folder+'/'+component)
            for i in range(len(images)):
                image = images[i]
                if train_val_ratio != 1:
                    img_type = 'val' if i > math.floor(len(images)*train_val_ratio) else 'train'
                else:
                    img_type = ''
                im = cv2.imread('./'+root_folder+'/'+component+'/'+image)
                cv2.imwrite('./'+root_folder+'_generated/images/'+img_type+'/'+component+'/'+image, im)
                bbs = get_bbs_from_image(im, clean=True, pad=30, text=True, combine_all=combine_all)
                bbs_str = '' 
                for bb in bbs:
                    bbs_str += str(labels[component])+' '+bb_to_str(normalize_bb(bb, im.shape[:2]))+'\n'
                    x,y,x2,y2 = pad_bb(bb, 5)
                    cv2.rectangle(im, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite('./'+root_folder+'_generated/images/bbs/'+image, im)
                f = open('./'+root_folder+'_generated/labels/'+image[:-3]+"txt", "a")
                f.write(bbs_str[:-1])
                f.close()
    else:
        images = os.listdir('./'+root_folder)
        for i in range(len(images)):
            image = images[i]
            if train_val_ratio != 1:
                img_type = 'val' if i > math.floor(len(images)*train_val_ratio) else 'train'
            else:
                img_type = ''
            im = cv2.imread('./'+root_folder+'/'+image)
            cv2.imwrite('./'+root_folder+'_generated/images/'+img_type+'/'+image, im)
            bbs = get_bbs_from_image(im, clean=True, pad=30, text=True, combine_all=combine_all)
            
            bbs_str = '' 
            c = 0
            for bb in bbs:
                c = c + 1
                bbs_str += str(c)+' '+bb_to_str(normalize_bb(bb, im.shape[:2]))+'\n'
                x,y,x2,y2 = pad_bb(bb, 5)
                cv2.rectangle(im, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(im, str(c), (int((x+x2)/2)-50,int((y+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5, cv2.LINE_AA)
            
            cv2.imwrite('./'+root_folder+'_generated/images/bbs/'+image, im)
            f = open('./'+root_folder+'_generated/labels/'+image[:-3]+"txt", "a")
            f.write(bbs_str[:-1])
            f.close()
            
# Removes predictions made which have a lower accuracy than the specified threshold
def threshold_output(prediction, threshold=0.5):
    output = {'boxes':[], 'scores':[], 'labels':[]}
    for i in range(len(prediction['scores'])):
        if prediction['scores'][i] > threshold:
            output['boxes'].append(prediction['boxes'][i])
            output['scores'].append(prediction['scores'][i])
            output['labels'].append(prediction['labels'][i])
    return output

# Creates a interpolation of the precision values utilized in calculating the mAP
def interpol_precision(precision, fptp):
    inter_prec = []
    curr_prec = precision[0]
    for i in range(len(precision)):
        if fptp[i]:
            curr_prec = precision[i]
        inter_prec.append(curr_prec)
    return inter_prec

# Calculates the Average Precision between a set of specified metrics by using the 11-point method
def calc_ap(precision, recall, fptp):
    inter_prec = interpol_precision(precision, fptp)
    AP = 0
    inter_prec.append(0)
    
    p = 0
    for i in range(0, 11):
        for c in range(p, len(recall)):
            if recall[c] < i*0.1:
                if p != len(recall):
                    p += 1
            else:
                break
        AP += inter_prec[p]
    
    return AP/11

# Calculates the Mean Average Precision using a model and a corresponding dataset
def calc_map(model, data_loader, device, num_classes, IoU=0.5):
    precision = [[] for k in range(0,num_classes)]
    recall = [[] for k in range(0,num_classes)]
    scores = [[] for k in range(0,num_classes)]
    fptp = [[] for k in range(0,num_classes)] # 0 = false positive, 1 = true positive
    fptp_p = [1 for k in range(0,num_classes)] # fptp pointer
    tpfn = [0 for k in range(0,num_classes)] # true positive false negative counter
    AP = [0 for k in range(0,num_classes)]
    mAP = 0
    predictions = []
    batch_nr = 0
    epoch_time = time.time()
    
    for images, targets in data_loader:
        batch_nr += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds = model(images)
        for i in range(len(preds)):
            thres = preds[i]
            for c in range(len(thres['labels'])):
                b_label = False
                for g in range(len(targets[i]['labels'])):
                    if device.type == 'cpu':
                        iou = calc_iou(targets[i]['boxes'][g].detach().numpy(), thres['boxes'][c].detach().numpy())
                    else:
                        iou = calc_iou(targets[i]['boxes'][g].cpu().detach().numpy(), thres['boxes'][c].cpu().detach().numpy())
                    if iou > IoU and thres['labels'][c].item() == targets[i]['labels'][g].item():
                        fptp[thres['labels'][c].item()].append(1)
                        scores[thres['labels'][c].item()].append(thres['scores'][c].item())
                        b_label = True
                        break;
                if not b_label:
                        fptp[thres['labels'][c].item()].append(0)
                        scores[thres['labels'][c].item()].append(thres['scores'][c].item())
                        tpfn[thres['labels'][c].item()] += 1

        for tar in targets:
            for lab in tar['labels']:
                tpfn[lab.item()] += 1
        
        print(
            '\r[Eval] mAP [{}/{}]\tEpoch time elapsed: {}'.format(
                batch_nr, len(data_loader), str(datetime.timedelta(seconds=round(time.time()-epoch_time)))
            ),
            end=''
        )
    
    
    
    t_fptp = copy.deepcopy(fptp)
    # Sort the arrays, highly inefficient sort O(n^2)
    for i in range(num_classes):
        for x in range(len(scores[i])):
            idx = -1
            highest_score = 0
            for c in range(len(scores[i])):
                if highest_score <= scores[i][c]:
                    highest_score = scores[i][c]
                    idx = c

            if idx != -1:
                t_fptp[i][x] = fptp[i][idx]
                scores[i][idx] = -1
            
    for i in range(len(t_fptp)):
        for c in range(1,len(t_fptp[i])+1):
            precision[i].append(sum(t_fptp[i][:c])/len(t_fptp[i][:c]))
            recall[i].append(sum(t_fptp[i][:c])/tpfn[i])
    
    for i in range(len(recall)):
        if precision[i] != []:
            AP[i] = calc_ap(precision[i], recall[i], t_fptp[i])*100
            mAP += AP[i]
    mAP = mAP/(num_classes-1)
    return AP[1:], mAP
            
# Creates a prediction of one or more images and saves the result in a specified location
def predict_and_save(model, root_dir, save_dir, labels=[], threshold=0.5, IoU=0, mask=False, unique_name='', skip_component=''):
    # Check if the 4th final character is a dot aka if the input directory is a file
    if root_dir[-4] == '.':
        im = root_dir
        img = cv2.imread(im)
        cv2_img = cv2.imread(im)
        if mask:
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
        tensor_img = torch.tensor(transform.to_tensor(img))
        tensor_img = torch.reshape(tensor_img, (1, tensor_img.size(0), tensor_img.size(1), tensor_img.size(2)))

        predictions = model(tensor_img)
        dont_print_id = []
        for i in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][i].item()
            if IoU > 0:
                if i in dont_print_id:
                    continue
                bb1 = predictions[0]['boxes'][i].detach().numpy()
                for c in range(i, len(predictions[0]['boxes'])):
                    bb2 = predictions[0]['boxes'][c].detach().numpy()
                    if calc_intersection(bb1, bb2) > calc_area(bb1)*IoU:
                        if labels[predictions[0]['labels'][i].item()-1] != skip_component and labels[predictions[0]['labels'][c].item()-1] == skip_component: 
                            dont_print_id.append(c)
            if score > threshold:
                x,y,x2,y2 = predictions[0]['boxes'][i].detach().numpy()
                cv2.rectangle(cv2_img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(cv2_img, str(score*100)[:5], (int((x+x2)/2)-200,int((y+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5, cv2.LINE_AA)
                
                if len(labels) > 1:
                    cv2.putText(cv2_img, labels[predictions[0]['labels'][i].item()-1], (int((x+x2)/2)-250,int((y+y2)/2)-150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)

        cv2.imwrite(save_dir+unique_name+(root_dir.split("/")[-1]), cv2_img)
    else:
        for image in os.listdir(root_dir):
            im = root_dir+image
            img = Image.open(im)
            cv2_img = cv2.imread(im)
            tensor_img = torch.tensor(transform.to_tensor(img))
            tensor_img = torch.reshape(tensor_img, (1, tensor_img.size(0), tensor_img.size(1), tensor_img.size(2)))

            predictions = model(tensor_img)
            for i in range(len(predictions[0]['boxes'])):
                score = predictions[0]['scores'][i].item()
                if score > threshold:
                    x,y,x2,y2 = predictions[0]['boxes'][i].detach().numpy()
                    cv2.rectangle(cv2_img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(cv2_img, str(score*100)[:5], (int((x+x2)/2)-200,int((y+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5, cv2.LINE_AA)

                    if len(labels) > 1:
                        cv2.putText(cv2_img, labels[predictions[0]['labels'][i].item()-1], (int((x+x2)/2)-250,int((y+y2)/2)-150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)

            cv2.imwrite(save_dir+unique_name+image, cv2_img)

# Saves the best bounding boxes from multiple predictions on the same image
# (Utilized for determening the component in the model council technique)
def save_best_bb(pred1, pred2, IoU=0.5):
    remove_pred1_ids = []
    remove_pred2_ids = []
    
    # Go through pred1 checking which elements to remove and also pred2
    for i in range(len(pred1['boxes'])):
        score1 = pred1['scores'][i].item()
        if pred1['boxes'][i].device.type == 'cpu':
            bb1 = pred1['boxes'][i].detach().numpy()
        else:
            bb1 = pred1['boxes'][i].cpu().detach().numpy()
        for c in range(len(pred2['boxes'])):
            score2 = pred2['scores'][c].item()
            
            if pred2['boxes'][c].device.type == 'cpu':
                bb2 = pred2['boxes'][c].detach().numpy()
            else:
                bb2 = pred2['boxes'][c].cpu().detach().numpy()
            intersect = calc_intersection(bb1, bb2)
            if intersect > calc_area(bb1)*IoU or intersect > calc_area(bb2)*IoU:
                if score1 > score2:
                    remove_pred2_ids.append(c)
                else:
                    remove_pred1_ids.append(i)
            
    # Remove the elements that should not be in the refined final list
    final_pred = {'boxes':[], 'scores':[], 'labels':[]}
    for i in range(len(pred1['boxes'])):
        if i not in remove_pred1_ids:
            final_pred['boxes'].append(pred1['boxes'][i])
            final_pred['scores'].append(pred1['scores'][i])
            final_pred['labels'].append(pred1['labels'][i])
    for i in range(len(pred2['boxes'])):
        if i not in remove_pred2_ids:
            final_pred['boxes'].append(pred2['boxes'][i])
            final_pred['scores'].append(pred2['scores'][i])
            final_pred['labels'].append(pred2['labels'][i])
    
    return final_pred
    
# Loads Faster R-CNN models based on the council technique
def frcnn_load_singular_models(model_name, components, root_dir):
    models = []
    device = torch.device('cpu')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for component in components:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.to(device)
        checkpoint = torch.load(root_dir+'/'+component+'/'+model_name+'.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
    return models

# Loads SSD models based on the council technique
def ssd_load_singular_models(model_name, components, root_dir, device):
    models = []

    for component in components:
        model_ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)

        num_classes = 2
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = [4, 6, 6, 6, 4, 4]
        model_ssd.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        
        model_ssd.to(device)
        
        checkpoint = torch.load(root_dir+'/'+component+'/'+model_name+'.pt')
        model_ssd.load_state_dict(checkpoint['model_state_dict'])
        model_ssd.eval()
        models.append(model_ssd)
        
    return models

# Creates a prediction of what text is located within the image
def predict_text(image_filename, spellcheck=True, raw_file=False, filtering=True):
    reader = easyocr.Reader(['en'],gpu = False) # load once only in memory.
    spell = SpellChecker()
    
    rotation_degrees = [0,1,-1]
    boxes = []
    words = []
    scores = []
    sharpen_kernel = numpy.array([[-1,-1,-1], [-1,15,-1], [-1,-1,-1]])
        
    for rotation in rotation_degrees:
        if not raw_file:
            image = cv2.imread(image_filename)
        else:
            image = image_filename
        image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        # sharp the edges or image.
        if filtering:
            if not raw_file:
                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
            else:
                sharpen = cv2.filter2D(rotated, -1, sharpen_kernel)
            thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            r_easy_ocr = reader.readtext(thresh)
        else:
            r_easy_ocr = reader.readtext(rotated)
        
        for pred in r_easy_ocr:
            box,word,score = pred
            x,y = box[0]
            x2,y2 = box[2]
            box = [int(x),int(y),int(x2),int(y2)]
            if len(boxes) < 1:
                boxes.append([box])
                words.append([word])
                scores.append([score])
            else:
                b_found = False
                for i in range(len(boxes)):
                    if calc_iou(box, boxes[i][0]) > 0.5 or calc_iou(boxes[i][0], box) > 0.5:
                        b_found = True
                        boxes[i].append(box)
                        words[i].append(word)
                        scores[i].append(score)
                        
                if not b_found:
                    boxes.append([box])
                    words.append([word])
                    scores.append([score])
    
    t_words = copy.deepcopy(words)
    t_boxes = copy.deepcopy(boxes)
    t_scores = copy.deepcopy(boxes)
    # Sort the arrays, highly inefficient sort O(n^2)
    for i in range(len(words)):
        for x in range(len(scores[i])):
            idx = -1
            highest_score = 0
            for c in range(len(scores[i])):
                if highest_score <= scores[i][c]:
                    highest_score = scores[i][c]
                    idx = c

            if idx != -1:
                t_words[i][x] = words[i][idx]
                t_boxes[i][x] = boxes[i][idx]
                t_scores[i][x] = scores[i][idx]
                scores[i][idx] = -1
                
    corr_words = []
    for i in range(len(t_words)):
        if spellcheck:
            corr_words.append(spell.correction(t_words[i][0]))
        else:
            corr_words.append(t_words[i][0])
        
    boxes = []
    for box in t_boxes:
        boxes.append(box[0])
        
    scores = []
    for score in t_scores:
        scores.append(score[0])

    return {'boxes':boxes, 'words':corr_words, 'scores':scores}

# Creates a prediction based on a specified model and image
def predict_model(model, image, IoU=0.5, disregard_comp=[], priority_comp=[]):
    model.eval()
    img = cv2.imread(image)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
    tensor_img = torch.tensor(transform.to_tensor(img))
    tensor_img = torch.reshape(tensor_img, (1, tensor_img.size(0), tensor_img.size(1), tensor_img.size(2)))
    
    pred = model(tensor_img)
    pred = {'boxes':pred[0]['boxes'].detach().numpy(), 'scores':pred[0]['scores'].detach().numpy(), 'labels':pred[0]['labels'].detach().numpy()}
    
    remove_arr = []
    for i in range(len(pred['boxes'])):
        for c in range(len(pred['boxes'])):
            if i == c:
                continue
            if calc_iou(pred['boxes'][i], pred['boxes'][c]) > IoU:
                if i in remove_arr or c in remove_arr:
                    continue
                
                in_disregard = pred['labels'][i] in disregard_comp or pred['labels'][c] in disregard_comp
                in_priority  = pred['labels'][i] in priority_comp  or pred['labels'][c] in priority_comp
                if not in_priority and in_disregard and pred['labels'][i] != pred['labels'][c]:
                    continue
                
                if not in_priority:
                    if pred['scores'][i] > pred['scores'][c]:
                        remove_arr.append(c)
                    else:
                        remove_arr.append(i)
                else:
                    if pred['labels'][i] in priority_comp:
                        remove_arr.append(c)
                    else:
                        remove_arr.append(i)

    remove_arr = list(dict.fromkeys(remove_arr))
    remove_arr.sort(reverse=True) # To get the highest id first
    
    new_boxes = pred['boxes']
    new_scores = pred['scores']
    new_labels = pred['labels']
    
    for idx in remove_arr:
        new_boxes  = numpy.delete(new_boxes, idx, 0)
        new_scores = numpy.delete(new_scores, idx, 0)
        new_labels = numpy.delete(new_labels, idx, 0)
        
    if len(remove_arr) > 0:
        return {'boxes':new_boxes, 'scores':new_scores, 'labels':new_labels}
    return pred

# Creates a prediction based on a council of models provided
def predict_models(models, images, IoU=0.5):
    predictions = [0 for k in range(len(images))]
    for i in range(len(images)):
        preds = []
        for model in models:
            pred = model([images[i]])
            preds.append(pred[0])

        for c in range(len(preds)):
            preds[c]['labels'] *= (c+1)

        final_pred = preds[0]
        for c in range(1, len(preds)):
            final_pred = save_best_bb(final_pred, preds[c], IoU=IoU)
        predictions[i] = final_pred
    return predictions
    
# Creates a prediction from a council of models and saves the results in a specified location
def predict_and_save_models(models, root_dir, save_dir, labels=[], threshold=0.5, IoU=0.5, mask=False):
    if root_dir[-4] == '.':
        im = root_dir
        img = cv2.imread(im)
        cv2_img = cv2.imread(im)
        if mask:
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
        tensor_img = torch.tensor(transform.to_tensor(img))
        tensor_img = torch.reshape(tensor_img, (1, tensor_img.size(0), tensor_img.size(1), tensor_img.size(2)))
        predictions = []
        for model in models:
            predictions.append(model(tensor_img)[0])

        for i in range(len(predictions)):
            predictions[i]['labels'] *= (i+1)
            
        final_pred = predictions[0]
        for i in range(1, len(predictions)):
            final_pred = save_best_bb(final_pred, predictions[i], IoU=IoU)
            
        for i in range(len(final_pred['boxes'])):
            score = final_pred['scores'][i].item()
            if score > threshold:
                x,y,x2,y2 = final_pred['boxes'][i].detach().numpy()
                cv2.rectangle(cv2_img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(cv2_img, str(score*100)[:5], (int((x+x2)/2)-200,int((y+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5, cv2.LINE_AA)

                if len(labels) > 1:
                    cv2.putText(cv2_img, labels[final_pred['labels'][i].item()-1], (int((x+x2)/2)-250,int((y+y2)/2)-150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)

        cv2.imwrite(save_dir+(root_dir.split("/")[-1]), cv2_img)
    else:
        for image in os.listdir(root_dir):
            im = root_dir+image
            img = Image.open(im)
            cv2_img = cv2.imread(im)
            tensor_img = transform.to_tensor(img)
            tensor_img = torch.reshape(tensor_img, (1, tensor_img.size(0), tensor_img.size(1), tensor_img.size(2)))

            predictions = []
            for model in models:
                predictions.append(model(tensor_img)[0])

            for i in range(len(predictions)):
                predictions[i]['labels'] *= (i+1)

            final_pred = predictions[0]
            for i in range(1, len(predictions)):
                final_pred = save_best_bb(final_pred, predictions[i], IoU=IoU)

            for i in range(len(final_pred['boxes'])):
                score = final_pred['scores'][i].item()
                if score > threshold:
                    x,y,x2,y2 = final_pred['boxes'][i].detach().numpy()
                    cv2.rectangle(cv2_img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(cv2_img, str(score*100)[:5], (int((x+x2)/2)-200,int((y+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5, cv2.LINE_AA)

                    if len(labels) > 1:
                        cv2.putText(cv2_img, labels[final_pred['labels'][i].item()-1], (int((x+x2)/2)-250,int((y+y2)/2)-150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)

            cv2.imwrite(save_dir+image, cv2_img)

# Creates a image which is a mix between two images based on a specified alpha
def mixup_data(x, y, alpha=1.0, use_cuda=False):
    batch_size = len(x)
    if (batch_size < 2):
        return x, [0], y
    
    # Get a random lambda
    if alpha > 0:
        lam = numpy.clip(numpy.random.beta(alpha, alpha), 0.4, 0.6)
    else:
        lam = 1

    # convert tensor array to numpy for mixup
    t_x = numpy.empty((batch_size, x[0].size()[1], x[0].size()[2]))
    for i in range(len(x)):
        t_x[i] = x[i].numpy().astype(numpy.float)
    
    # convert tuple to numpy array for easier indexing
    t_y = numpy.empty((batch_size), dtype=numpy.object)
    for i in range(len(y)):
        t_y[i] = {}
        for var in y[i]:
            t_y[i][var] = y[i][var].numpy().astype(numpy.double)
        
    # Get a random set of indicies
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    # Mix the two images and make them % transparent based on lambda
    t_mixed_x = lam * t_x + (1 - lam) * t_x[index, :]
    y_a, y_b = t_y, t_y[index]

    # Zip together the bounding boxes of the zipped images
    mixedup_bboxes = []
    i = 0
    for bbox, s_bbox in zip(y_a, y_b):
        # If the two images zipped are the same, keep one of the boundingbox infos
        if (bbox['boxes'][0] == s_bbox['boxes'][0]).all():
            mix_len = len(bbox['boxes'])
            mixedup_bboxes.append({'boxes':torch.zeros([mix_len, 4], dtype=torch.double), 'labels':torch.zeros([mix_len], dtype=torch.int64), 'image_id':torch.zeros([1], dtype=torch.int64), 'area':torch.zeros([mix_len], dtype=torch.float), 'iscrowd':torch.zeros([mix_len], dtype=torch.int64)})
            for c in range(len(bbox['boxes'])):
                mixedup_bboxes[i]['boxes'][c] = torch.from_numpy(bbox['boxes'][c])
                mixedup_bboxes[i]['labels'][c] = bbox['labels'][c]
                mixedup_bboxes[i]['area'][c] = bbox['area'][c]
                mixedup_bboxes[i]['iscrowd'][c] = bbox['iscrowd'][c]
            i += 1
            continue;
            
        mix_len = len(bbox['boxes'])+len(s_bbox['boxes'])
        mixedup_bboxes.append({'boxes':torch.zeros([mix_len, 4], dtype=torch.double), 'labels':torch.zeros([mix_len], dtype=torch.int64), 'image_id':torch.zeros([1], dtype=torch.int64), 'area':torch.zeros([mix_len], dtype=torch.float), 'iscrowd':torch.zeros([mix_len], dtype=torch.int64)})
        for c in range(len(bbox['boxes'])):
            mixedup_bboxes[i]['boxes'][c] = torch.from_numpy(bbox['boxes'][c])
            mixedup_bboxes[i]['labels'][c] = bbox['labels'][c]
            mixedup_bboxes[i]['area'][c] = bbox['area'][c]
            mixedup_bboxes[i]['iscrowd'][c] = bbox['iscrowd'][c]
        
        for j in range(len(s_bbox['boxes'])):
            mixedup_bboxes[i]['boxes'][c+j+1] = torch.from_numpy(s_bbox['boxes'][j])
            mixedup_bboxes[i]['labels'][c+j+1] = s_bbox['labels'][j]
            mixedup_bboxes[i]['area'][c+j+1] = s_bbox['area'][j]
            mixedup_bboxes[i]['iscrowd'][c+j+1] = s_bbox['iscrowd'][j]
            
        mixedup_bboxes[i]['image_id'][0] = bbox['image_id'][0]
        i += 1
        
    mixed_x = []
    for v in t_mixed_x:
        mixed_x.append(torch.FloatTensor([v]))
    
    return mixed_x, index, tuple(mixedup_bboxes)

# Generates new combined data based on how many components it can fit into the screen without overlapping too much
def generate_combined_data(root_folder, max_nr_components=3, IoU=0.05, start_index_filename=0, component_folder=''):
    os.mkdir('./'+root_folder+'_combined_'+str(max_nr_components)+'/')
    os.mkdir('./'+root_folder+'_combined_'+str(max_nr_components)+'/images/')
    os.mkdir('./'+root_folder+'_combined_'+str(max_nr_components)+'/labels/')
    components = os.listdir('./'+root_folder+'/images')
    max_nr_components -= 1

    if component_folder == '':
        component_folder = root_folder
        
    images = []
    components = os.listdir('./'+component_folder+'/images')
    for component in components:
        images.append(os.listdir('./'+component_folder+'/images/'+component))

    root_images = []
    root_components = os.listdir('./'+root_folder+'/images')    
    for component in root_components:
        root_images.append(os.listdir('./'+root_folder+'/images/'+component))

    for i in range(len(root_components)):
        print('Making dataset for:',root_components[i])
        for img in root_images[i]:
            image = []
            labels = []
            im = cv2.imread('./'+root_folder+'/images/'+root_components[i]+'/'+img)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)

            f = open('./'+root_folder+'/labels/'+img[:-3]+'txt', 'r')
            data = f.read().split('\n')
            f.close()
            for line in data:
                labels.append(str_to_bb(line))

            nr_components = 0
            components_checked = []
            for c in range(len(images)):
                if max_nr_components == nr_components:
                    break;

                choices = []
                for y in range(len(components)):
                    if not y in components_checked:
                        choices.append(y)
                if len(choices) == 0:
                    break;
                c = numpy.random.choice(choices)
                components_checked.append(c)
                
                idxs = [y for y in range(len(images[c]))]
                random.shuffle(idxs)
                for y in idxs:
                    f = open('./'+component_folder+'/labels/'+images[c][y][:-3]+'txt', 'r')
                    data = f.read().split('\n')
                    f.close()

                    intersects = False
                    new_labels = labels.copy()
                    for line in data:
                        line_bb = str_to_bb(line)
                        denorm_line_bb = denormalize_bb(line_bb[1:], image.shape[:2])
                        for bb in labels:
                            denorm_bb = denormalize_bb(bb[1:], image.shape[:2])
                            if calc_iou(denorm_bb, denorm_line_bb) > IoU or calc_iou(denorm_line_bb, denorm_bb) > IoU:
                                intersects = True
                                break;
                        if intersects:
                            break;
                        new_labels.append(line_bb)
                    if intersects:
                        continue;

                    labels = new_labels.copy()
                    im2 = cv2.imread('./'+component_folder+'/images/'+components[c]+'/'+images[c][y])
                    imgray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
                    ret, image2 = cv2.threshold(imgray2, 100, 255, cv2.THRESH_BINARY_INV)
                    image = image+image2
                    nr_components += 1
                    break;

            str_labels = ''
            for bb in labels:
                str_labels += str(int(bb[0]))+' '+bb_to_str(bb[1:])+'\n'
            cv2.imwrite('./'+root_folder+'_combined_'+str(max_nr_components+1)+'/images/'+str(start_index_filename)+'.jpg', image)
            f = open('./'+root_folder+'_combined_'+str(max_nr_components+1)+'/labels/'+str(start_index_filename)+'.txt', 'w')
            f.write(str_labels[:-1])
            f.close()
            start_index_filename += 1
            
# Helper function for utilizing SketchDataset
def collate_fn(batch):
    return tuple(zip(*batch))

# A dataset class for supporting the custom datasets utilized
class SketchDataset(Dataset):
    # An initialize function which requires location of the dataset and meta data regarding the dataset at hand
    def __init__(self, root_dir, set_type, single_component=False, combined=False, preprocessed=False, only_component_id=-1, text_label=False):
        self.images = []
        self.labels = []
        self.component_names = []
        if combined:
            for image in os.listdir(root_dir+"/images/"+set_type):
                self.images.append(root_dir+"/images/"+set_type+'/'+image)
                self.labels.append(root_dir+"/labels/"+(image.split('.')[0]+'.txt'))
        elif not single_component:
            for component in os.listdir(root_dir+"/images/"+set_type):
                if component == 'Combined':
                    continue
                self.component_names.append(component)
                for image in os.listdir(root_dir+"/images/"+set_type+"/"+component):
                    self.images.append(root_dir+"/images/"+set_type+"/"+component+"/"+image)
                    self.labels.append(root_dir+"/labels/"+(image.split('.')[0]+'.txt'))
        else:
            self.component_names.append(single_component)
            for image in os.listdir(root_dir+"/images/"+set_type+"/"+single_component):
                self.images.append(root_dir+"/images/"+set_type+"/"+single_component+"/"+image)
                self.labels.append(root_dir+"/labels/"+(image.split('.')[0]+'.txt'))
            
        self.root = root_dir
        self.single_component = single_component
        self.combined = combined
        self.preprocessed = preprocessed
        self.comp_id = only_component_id
        self.text_label = text_label
        
    def __len__(self):
        return len(self.images)
    
    # Returns the image with its corresponding labels
    def __getitem__(self, idx):
        im = cv2.imread(self.images[idx])
        if not self.preprocessed:
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
        else:
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        f = open(self.labels[idx], "r")
        data = f.read().split('\n')
        f.close()

        N = 0
        for i in range(len(data)):
            if self.comp_id == -1:
                N += 1
            else:
                curr_id = int(data[i].split(' ')[0])
                if self.comp_id == curr_id:
                    N += 1
                    
        boxes = torch.zeros([N, 4], dtype=torch.double)
        if self.text_label:
            labels = [0 for i in range(N)]
        else:
            labels = torch.zeros([N], dtype=torch.int64)
        areas = torch.zeros([N])
        
        for i in range(N):
            if self.text_label:
                word = data[i].split(' ')[0]
                data[i] = ' '.join(['0']+data[i].split(' ')[1:])
            bb = denormalize_bb(str_to_bb(data[i])[1:], img.shape[:2])
            boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3] = bb
            areas[i] = calc_area(bb)
            
            if not self.single_component and self.comp_id == -1:
                if self.text_label:
                    labels[i] = word
                else:
                    labels[i] = int(data[i].split(' ')[0])+1
                continue
                
            labels[i] = 1
                
        return transform.to_tensor(img), {'boxes':boxes, 'labels':labels, 'image_id':torch.LongTensor([idx]), 'area':areas, 'iscrowd':torch.zeros([N], dtype=torch.int64)}

# Trains a specified model with a specified data set for a defined number of epochs
def train_model(model, optimizer, data_loader, data_loader_val, device, num_epochs, model_type, model_name, lr_scheduler=False, folder_name='', mixup=False, begin_epoch=0):
    writer = SummaryWriter()
    total_time = time.time()
    
    if folder_name == '':
        folder_name = datetime.datetime.now().strftime("%b-%d_%H-%M")
    
    if not os.path.exists('./models/'+model_type+'/'+folder_name+'/'):
        os.mkdir('./models/'+model_type+'/'+folder_name+'/')

    for epoch in range(begin_epoch, num_epochs):
        epoch_time = time.time()
        epoch_loss = []
        batch_nr = 0
        
        for images, targets in data_loader:
            batch_time = time.time()
            if mixup:
                images, _, targets = mixup_data(images, targets)

            # Send them to device if using GPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            pred = model(images, targets)
            losses = sum(loss for loss in pred.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss.append(losses.item())
            
            writer.add_scalars(model_type+'_'+model_name, {
                'train_loss': losses.item(),
            }, epoch*len(data_loader)+batch_nr)
            
            batch_nr = batch_nr + 1
            print_loss = losses.item()
            
            if batch_nr == epoch+1:
                print_loss = numpy.average(epoch_loss)
                
            print(
                '\r[Train] Epoch {} [{}/{}] - Loss: {} \tProgress [{}%] \tEpoch time elapsed: {}'.format(
                    epoch+1, batch_nr, len(data_loader), print_loss, round(((epoch/num_epochs)+(1/num_epochs*batch_nr/len(data_loader)))*100, 2), str(datetime.timedelta(seconds=round(time.time()-epoch_time)))
                ),
                end=''
            )
        
            
        writer.add_scalars(model_type+'_'+model_name, {
            'avg_epoch_loss': numpy.average(epoch_loss),
        }, (epoch+1))
            
        if lr_scheduler:
            lr_scheduler.step()
        
        print()
        evaluate_model(model, data_loader_val, device, writer, model_type, model_name, epoch+1)
        #model.train()
        print()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './models/'+model_type+'/'+folder_name+'/'+model_name+'-'+str(epoch+1)+'.pt')

    print(
        '\rTraining completed! Loss: {} \tTotal time elapsed: {}'.format(
            losses.item(), str(datetime.timedelta(seconds=round(time.time()-total_time)))
        ),
        end=''
    )
    
# Evaluates a model based on a evaluation dataset
def evaluate_model(model, data_loader, device, writer, model_type, model_name, epoch):
    with torch.no_grad():
        epoch_time = time.time()
        avg_loss = []
        batch_nr = 0
        for images, targets in data_loader:
            # Send them to device if using GPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            pred = model(images, targets)
            losses = sum(loss for loss in pred.values())
            avg_loss.append(losses.item())
            
            batch_nr = batch_nr + 1
            print_loss = losses.item()
            
            if batch_nr == epoch+1:
                print_loss = numpy.average(avg_loss)
            print(
                '\r[Val] [{}/{}] - Loss: {} \tEpoch time elapsed: {}'.format(
                    batch_nr, len(data_loader), print_loss, str(datetime.timedelta(seconds=round(time.time()-epoch_time)))
                ),
                end=''
            )

        writer.add_scalars(model_type+'_'+model_name, {
            'val_loss': numpy.average(avg_loss),
        }, epoch)

# A function for training a council of Faster R-CNN models
def train_multi_frcnn(model_name, model_type, epochs, components=[]):
    if components == []:
        f = open('./dataset/labels.txt', "r")
        data = f.read().split('\n')
        f.close()
        components = {data[i]:i for i in range(len(data))}

    for component in components:
        dataset_train = SketchDataset('./dataset', 'train', single_component=component)
        dataset_val = SketchDataset('./dataset', 'val', single_component=component)
        data_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=5, shuffle=True, num_workers=0,
                collate_fn=collate_fn)
        data_loader_val = torch.utils.data.DataLoader(
                dataset_val, batch_size=2, shuffle=False, num_workers=0,
                collate_fn=collate_fn)

        device = torch.device('cpu')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, 
                                    momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        train_model(model, optimizer, data_loader, data_loader_val, device, epochs, model_type, model_name, lr_scheduler, folder_name=component+'_'+model_name)

# A function for training a council of SSD models
def train_multi_ssd(model_name, model_type, epochs, components=[]):
    if components == []:
        f = open('./dataset/labels.txt', "r")
        data = f.read().split('\n')
        f.close()
        components = {data[i]:i for i in range(len(data))}

    for component in components:
        print('Training:',component)
        print()
        dataset_train = SketchDataset('./dataset', 'train', single_component=component)
        dataset_val = SketchDataset('./dataset', 'val', single_component=component)
        data_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=5, shuffle=True, num_workers=0,
                collate_fn=collate_fn)
        data_loader_val = torch.utils.data.DataLoader(
                dataset_val, batch_size=2, shuffle=False, num_workers=0,
                collate_fn=collate_fn)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model_ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)

        num_classes = 2
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = [4, 6, 6, 6, 4, 4]
        model_ssd.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)


        model_ssd.to(device)


        params = [p for p in model_ssd.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.00005, 
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)
        

        train_model(model_ssd, optimizer, data_loader, data_loader_val, device, epochs, model_type, model_name, lr_scheduler, folder_name=component+'_'+model_name)
    
# Loads a Faster R-CNN model from a specified location
def load_frcnn(date, model_name, num_classes=13):
    device = torch.device('cpu')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, 
                                momentum=0.9, weight_decay=0.0005)

    checkpoint = torch.load('./models/Faster-RCNN/'+date+'/'+model_name+'.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model,optimizer
    
# Loads a SSD model from a specified location
def load_ssd(date, model_name):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    num_classes = 13
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    model_ssd.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    model_ssd.to(device)

    params = [p for p in model_ssd.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, 
                                momentum=0.9, weight_decay=0.0005)

    checkpoint = torch.load('./models/SSD/'+date+'/'+model_name+'.pt', map_location=device)
    model_ssd.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model_ssd,optimizer,device

# Post-processes the results provided according to the specifications in the thesis
def post_process_results(results, comp_defs, labels, min_score=[0.5, 0.5], non_ui_components=[10,11,12], debug_print=False):
    bb_res,txt_res = results
    
    reference_id      = non_ui_components[0]
    object_id         = non_ui_components[1]
    reference_head_id = non_ui_components[2]

    if debug_print:
        print('Saniticing results')
    # 1. First sanitice the results aka remove anything less than min_score
    remove_results = []
    remove_txt    = []
    for i in range(len(bb_res['boxes'])):
        if bb_res['scores'][i] < min_score[0]:
            remove_results.append(i)
            
    for i in range(len(txt_res['boxes'])):
        if len(txt_res['words'][i]) < 2:
            remove_txt.append(i)
            continue
            
        if txt_res['scores'][i] < min_score[1]:
            remove_txt.append(i)
        else:
            for c in range(len(bb_res['boxes'])):
                # If we have a larger text than the object it intersects with we want to remove it
                # This is because words can be interpereted as components in the sketch
                txt_box = txt_res['boxes'][i].copy()
                if bb_res['labels'][c] in [8,9]:
                    txt_box[0] += 200
                fully_overlap = calc_overlap(bb_res['boxes'][c], txt_box) > 0.75
                if fully_overlap and c not in remove_results:
                    remove_results.append(c)

    remove_results.sort(reverse=True) # To get the highest id first
    remove_txt.sort(reverse=True) # To get the highest id first
    
    saniticed_bb_results = copy.deepcopy(bb_res)
    for idx in remove_results:
        if debug_print:
            print('Removing:', labels[int(saniticed_bb_results['labels'][idx])-1])
        saniticed_bb_results['boxes']  = numpy.delete(saniticed_bb_results['boxes'], idx, 0)
        saniticed_bb_results['scores'] = numpy.delete(saniticed_bb_results['scores'], idx, 0)
        saniticed_bb_results['labels'] = numpy.delete(saniticed_bb_results['labels'], idx, 0)
        
    saniticed_txt_results = copy.deepcopy(txt_res)
    for idx in remove_txt:
        saniticed_txt_results['boxes']  = numpy.delete(saniticed_txt_results['boxes'], idx, 0)
        saniticed_txt_results['scores'] = numpy.delete(saniticed_txt_results['scores'], idx, 0)
        saniticed_txt_results['words'] = numpy.delete(saniticed_txt_results['words'], idx, 0)
        
    if debug_print:
        print('Removed', len(remove_results), 'sketch objects and',len(remove_txt),'text objects')

    word_to_idx = {}
    len_before_add = len(saniticed_bb_results['boxes'])
    # 1.1 Check if any string doesnt overlap at for a new component to be created
    for i in range(len(saniticed_txt_results['boxes'])):
        t_found_overlap = False
        for c in range(len(saniticed_bb_results['boxes'])):
            paragraphBigger = calc_iou(saniticed_txt_results['boxes'][i], saniticed_bb_results['boxes'][c]) < calc_iou(saniticed_bb_results['boxes'][c], saniticed_txt_results['boxes'][i])
            if not paragraphBigger and calc_iou(saniticed_txt_results['boxes'][i], saniticed_bb_results['boxes'][c]) > 0.2:
                t_found_overlap = True
                break
    
        if not t_found_overlap:
            saniticed_bb_results['boxes'] = numpy.append(saniticed_bb_results['boxes'], [saniticed_txt_results['boxes'][i]], axis = 0)
            saniticed_bb_results['scores'] = numpy.append(saniticed_bb_results['scores'], [saniticed_txt_results['scores'][i]], axis = 0)
            saniticed_bb_results['labels'] = numpy.append(saniticed_bb_results['labels'], [saniticed_txt_results['words'][i]], axis = 0)            
            word_to_idx[calc_area(saniticed_bb_results['boxes'][len(saniticed_bb_results['boxes'])-1])] = saniticed_txt_results['words'][i]
            if debug_print:
                print('Words added:',saniticed_txt_results['words'][i])
    
    if debug_print:
        print('Added',len(saniticed_bb_results['boxes'])-len_before_add, 'text components')
    
    # 2. Resolve which texts are paragraphs and which are additions to a certain component
    remove_txt = []
    for i in range(len(saniticed_txt_results['boxes'])):
        t_cover_percentage = []
        t_found_component = False
        t_self_idx = -1
        for c_def in comp_defs[len(comp_defs)-1][4]:
            for c in range(len(saniticed_bb_results['boxes'])):
                txt_box = saniticed_txt_results['boxes'][i].copy()
                txt_box[0] -= 100
                curr_iou = calc_iou(txt_box, saniticed_bb_results['boxes'][c])
                if saniticed_bb_results['labels'][c] == saniticed_txt_results['words'][i]:
                    t_self_idx = c
                    if not t_found_component:
                        continue
                    else:
                        break
                        
                if t_found_component:
                    continue;
                    
                if curr_iou > 0.0:
                    t_cover_percentage.append((c, curr_iou))
                    try:
                        lbl = int(saniticed_bb_results['labels'][c])
                    except ValueError:
                        lbl = 13
                    if lbl-1 != c_def:
                        continue;
                    if comp_defs[lbl-1][0] or comp_defs[lbl-1][2]:
                        word_to_idx[calc_area(saniticed_bb_results['boxes'][c])] = saniticed_txt_results['words'][i]
                        if debug_print:
                            print('Found text for a component: ',labels[lbl-1], '->', saniticed_txt_results['words'][i])
                        t_found_component = True
                        if t_self_idx != -1:
                            break
                        
            if t_found_component:
                if t_self_idx != -1:
                    remove_txt.append(t_self_idx)
                break;   
        
        if not t_found_component and len(t_cover_percentage) > 0:
            t_cover_percentage.sort(key=lambda y: y[1], reverse=True)
            t_changed_component = False
            for idx,iou in t_cover_percentage:
                try:
                    idx_lbl = int(saniticed_bb_results['labels'][idx])-1
                except ValueError:
                    continue
                for c_def in comp_defs[idx_lbl][4]:
                    if comp_defs[c_def][0] or comp_defs[c_def][2]:
                        saniticed_bb_results['labels'][idx] = c_def+1
                        word_to_idx[calc_area(saniticed_bb_results['boxes'][idx])] = saniticed_txt_results['words'][i]
                        if debug_print:
                            print('Found text overlapping a unsupported component, component changed from: ',labels[idx_lbl], 'to', labels[c_def], '->', saniticed_txt_results['words'][i])
                        t_changed_component = True
                        break;
                if t_changed_component:
                    break;
                        
                
    
    remove_txt.sort(reverse=True) # To get the highest id first
    for idx in remove_txt:
        saniticed_bb_results['boxes']  = numpy.delete(saniticed_bb_results['boxes'], idx, 0)
        saniticed_bb_results['scores'] = numpy.delete(saniticed_bb_results['scores'], idx, 0)
        saniticed_bb_results['labels'] = numpy.delete(saniticed_bb_results['labels'], idx, 0)

    # 3. Resolve references
    references      = []
    reference_heads = []
    objects         = []
    for i in range(len(saniticed_bb_results['boxes'])):
        try:
            if int(saniticed_bb_results['labels'][i]) == reference_id:
                references.append(i)
            elif int(saniticed_bb_results['labels'][i]) == reference_head_id:
                reference_heads.append(i)
            elif int(saniticed_bb_results['labels'][i]) == object_id:
                objects.append(i)
        except ValueError:
            continue
            
    # If the model has found the heads to the reference we can easily deduce its direction otherwise we need to deduce it manually
    ref_to_obj = []
    if references == [] and reference_heads != []:
        references = reference_heads.copy()
        reference_heads = []
        
    if reference_heads != []:
        for i, ref_idx in enumerate(references):
            box = saniticed_bb_results['boxes'][ref_idx]
            head_box = saniticed_bb_results['boxes'][reference_heads[i]]
            p1_ul = [box[0],box[1]]
            p1_ur = [box[2],box[1]]
            p1_ll = [box[0],box[3]]
            p1_lr = [box[2],box[3]]
            p1 = [p1_ul, p1_ur, p1_ll, p1_lr]
            
            head_p = [(head_box[2]-head_box[0])/2+head_box[0], (head_box[3]-head_box[1])/2+head_box[1]]
            smallest_distance = 10000000
            point = head_p
            oppo_point = (-1, -1)
            
            for c, ref_p in enumerate(p1):
                diff = ((((ref_p[0] - head_p[0] )**2) + ((ref_p[1] - head_p[1])**2) )**0.5)
                if diff < smallest_distance:
                    smallest_distance = diff
                    oppo_point = p1[3-c]
            
            smallest_distance = 10000000  
            idx = -1
            
            for obj_idx in objects:
                box = saniticed_bb_results['boxes'][obj_idx]
                p2_ul = [box[0],box[1]]
                p2_ur = [box[2],box[1]]
                p2_ll = [box[0],box[3]]
                p2_lr = [box[2],box[3]]
                p2 = [p2_ul, p2_ur, p2_ll, p2_lr]
                for obj_p in p2:                    
                    diff = ((((obj_p[0] - head_p[0] )**2) + ((obj_p[1] - head_p[1])**2) )**0.5)
                    if diff < smallest_distance:
                        smallest_distance = diff
                        idx = obj_idx
                            
            if idx > -1:
                ref_to_obj.append((ref_idx, idx, point, oppo_point))
    else:
        for ref_idx in references:
            box = saniticed_bb_results['boxes'][ref_idx]
            p1_ul = [box[0],box[1]]
            p1_ur = [box[2],box[1]]
            p1_ll = [box[0],box[3]]
            p1_lr = [box[2],box[3]]
            p1 = [p1_ul, p1_ur, p1_ll, p1_lr]
            smallest_distance = 10000000
            idx = -1
            point = (-1, -1)
            oppo_point = (-1, -1)

            for obj_idx in objects:
                box = saniticed_bb_results['boxes'][obj_idx]
                p2_ul = [box[0],box[1]]
                p2_ur = [box[2],box[1]]
                p2_ll = [box[0],box[3]]
                p2_lr = [box[2],box[3]]
                p2 = [p2_ul, p2_ur, p2_ll, p2_lr]
                for ref_p in p1:
                    for obj_p in p2:
                        diff = ((((obj_p[0] - ref_p[0] )**2) + ((obj_p[1] - ref_p[1])**2) )**0.5)
                        
                        if diff < smallest_distance:
                            smallest_distance = diff
                            idx = obj_idx
                            point = ref_p
                            oppo_point = p1[3-p1.index(ref_p)]
                            
            if idx > -1:
                ref_to_obj.append((ref_idx, idx, point, oppo_point))
    
    if debug_print:
        print('Attempting to resolve references')
    for i in range(len(ref_to_obj)):
        p = ref_to_obj[i][3]
        smallest_distance = 10000000
        comp_idx = -1
        smallest_point = (-1,-1)
        for c in range(len(saniticed_bb_results['boxes'])):
            try:
                comp_lbl = int(saniticed_bb_results['labels'][c])
            except ValueError:
                comp_lbl = 13
                
            if comp_lbl == reference_id or comp_lbl == reference_head_id or comp_lbl == object_id:
                continue
                
            box = saniticed_bb_results['boxes'][c]
            p2_ul = [box[0],box[1]]
            p2_ur = [box[2],box[1]]
            p2_ll = [box[0],box[3]]
            p2_lr = [box[2],box[3]]
            p2 = [p2_ul, p2_ur, p2_ll, p2_lr]
            for comp_p in p2:
                diff = ((((comp_p[0] - p[0] )**2) + ((comp_p[1] - p[1])**2) )**0.5)
                if diff < smallest_distance:
                    smallest_distance = diff
                    comp_idx = c
                    smallest_point = comp_p
    
        if comp_idx > -1:
            try:
                comp_lbl = int(saniticed_bb_results['labels'][comp_idx])-1
            except ValueError:
                comp_lbl = 12

            if comp_defs[comp_lbl][1] or comp_defs[comp_lbl][3]:
                ref_to_obj[i] = (comp_idx, ref_to_obj[i][1], ref_to_obj[i][2], ref_to_obj[i][3])
            else:
                if debug_print:
                    print('Did not find a suitable comp directly, attempting to change closest component')
                t_found_other = False
                for other_comp in comp_defs[comp_lbl][4]:
                    if comp_defs[other_comp][1] or comp_defs[other_comp][3]:
                        word_to_idx[calc_area(saniticed_bb_results['boxes'][comp_idx])] = saniticed_bb_results['labels'][comp_idx]
                        saniticed_bb_results['labels'][comp_idx] = other_comp+1
                        ref_to_obj[i] = (comp_idx, ref_to_obj[i][1], ref_to_obj[i][2], ref_to_obj[i][3])
                        t_found_other = True
                        break

                if not t_found_other:
                    ref_to_obj[i] = (-1, ref_to_obj[i][1], ref_to_obj[i][2], ref_to_obj[i][3])
                    # Since we did not find a comp that supported references for the closest comp
                    # we define the reference as undefined
                    
    # Clean up word_to_idx, since we used it more as word_to_size
    for i in range(len(saniticed_bb_results['boxes'])):
        try:
            word = word_to_idx[calc_area(saniticed_bb_results['boxes'][i])]
            word_to_idx[i] = word
            del word_to_idx[calc_area(saniticed_bb_results['boxes'][i])]
        except KeyError:
            continue

    # 4. Sort the array by the biggest objects
    idx_to_size = []
    for i in range(len(saniticed_bb_results['boxes'])):
        idx_to_size.append((i, calc_area(saniticed_bb_results['boxes'][i])))
    idx_to_size.sort(key=lambda y: y[1], reverse=True)
     
    # 5. Top-down walk to get the row/cols of the layout
    rows = []
    ignore_comps = []
    for c in range(len(saniticed_bb_results['boxes'])):
        top_element = [0, 1000000, 0, 0]
        top_element_idx = -1
        # Get the current highest element from the picture
        for i,box in enumerate(saniticed_bb_results['boxes']):
            if i in ignore_comps:
                continue;
            try:
                if int(saniticed_bb_results['labels'][i]) in non_ui_components:
                    continue;
            except ValueError:
                pass
            if box[1] < top_element[1]:
                top_element[0] = box[0]
                top_element[1] = box[1]
                top_element[2] = box[2]
                top_element[3] = box[3]
                top_element_idx = i
        if top_element_idx == -1:
            break;
        cols = [(top_element_idx, top_element[0])]
        # Make so that we cover all the sides of the top element to find anything on that row
        top_element[0] = 0
        top_element[2] = 1000000
        # Now we need to find which elements are on the same row as the current highest element
        for i,box in enumerate(saniticed_bb_results['boxes']):
            if i == top_element_idx:
                continue;
            if i in ignore_comps:
                continue;
            try:
                if int(saniticed_bb_results['labels'][i]) in non_ui_components:
                    continue;
            except ValueError:
                pass
            if calc_iou(top_element, box) > 0:
                cols.append((i, box[0]))
        
        # Now we got the objects on the same column, now we need to find the biggest of that intersects with a larger portion
        for size in idx_to_size:
            biggest_idx = -1
            for col in cols:
                if col[0] == size[0]:
                    biggest_idx = size[0]
                    break;
            if biggest_idx != -1:
                break;
        biggest_cols = []
        biggest_box = numpy.copy(saniticed_bb_results['boxes'][biggest_idx])
        biggest_box[0] = 0
        biggest_box[2] = 1000000
        for i,box in enumerate(saniticed_bb_results['boxes']):
            if i in ignore_comps:
                continue;
            try:
                if int(saniticed_bb_results['labels'][i]) in non_ui_components:
                    continue;
            except ValueError:
                pass
            if calc_iou(biggest_box, box) > 0:
                biggest_cols.append((i, box))
        all_cols = []
        for col in biggest_cols:            
            curr_col = []
            curr_box = [0, col[1][1], 1000000, col[1][3]]
            for i,box in enumerate(saniticed_bb_results['boxes']):
                if i in ignore_comps:
                    continue;
                try:
                    if int(saniticed_bb_results['labels'][i]) in non_ui_components:
                        continue;
                except ValueError:
                    pass
                if calc_iou(curr_box, box) > 0:
                    curr_col.append((i, box))
            if curr_col != []:
                all_cols.append(curr_col)
            else:
                if debug_print:
                    print("ERROR, THIS SHOULD NOT BE REACHED! THIS IS SUPER STRANGE SINCE WE SHOULD ALWAYS GET A OVERLAP")

        if len(all_cols) > 1:
            remove_idxs = [] # This is because we want to remove entries with more columns than it actually is
            longest_intersect_col = 0
            
            for i in range(len(all_cols)):
                len_of_col = 1000
                for c in range(len(all_cols[i])):
                    curr_box = [0, all_cols[i][c][1][1], 1000000, all_cols[i][c][1][3]]
                    overlap = 0
                    for y in range(len(all_cols[i])):
                        if calc_iou(curr_box, all_cols[i][y][1]) > 0:
                            overlap += 1
                    if overlap < len_of_col:
                        len_of_col = overlap
                        
                if len_of_col < len(all_cols[i]):
                    remove_idxs.append(i)
                if len_of_col > longest_intersect_col:
                    longest_intersect_col = len_of_col

            remove_idxs.sort(reverse=True)
            for idx in remove_idxs:
                all_cols.pop(idx)
                
            new_cols = []
            for c in range(len(all_cols)):
                curr_col = all_cols[c]
                curr_col.sort(key=lambda y: y[1][0])
                sorted_col = []
                for t_col in curr_col:
                    sorted_col.append((t_col[0], t_col[1][0], t_col[1][1]))
                new_cols.append(sorted_col)
                
            b_same = True
            for col1 in new_cols:
                for col2 in new_cols:
                    if col1 != col2:
                        b_same = False
                        break;
                if not b_same:
                    break;
            if b_same:
                rows.append(new_cols[0])
            else:        
                remove_idxs = []
                for idx, col in enumerate(new_cols):
                    except_idx = new_cols[0:idx]+new_cols[idx+1:len(new_cols)]
                    if col in except_idx and not except_idx.index(col) in remove_idxs and not idx in remove_idxs:
                        remove_idxs.append(idx)
                
                remove_idxs.sort(reverse=True)
                for idx in remove_idxs:
                    new_cols.pop(idx)
                    
                rows.append((True, new_cols))
        else:
            a_cols = all_cols[0]
            fixed_col = []
            for t_col in a_cols:
                fixed_col.append((t_col[0], t_col[1][0], t_col[1][1]))
            rows.append(fixed_col)
        
        for acol in all_cols:
            for col in acol:
                ignore_comps.append(col[0])

    # 6. Create json results
    if debug_print:
        print('Creating json formating')
    json_dict = {"rows":{}, "objects":{}}
    for idx,ref in enumerate(ref_to_obj):
        json_dict["objects"][str(idx)] = word_to_idx[ref[1]]
    
    for row_idx,row in enumerate(rows):
        row_dict = {}
        if type(row) is tuple:
            # Multiple rows in same row
            largest_col = 0
            multi_rows = row[1]
            for mrow in multi_rows:
                if len(mrow) > largest_col:
                    largest_col = len(mrow)
            columns = []
            for col_idx in range(largest_col):
                components = []
                for mrow_idx in range(len(multi_rows)):                    
                    try:
                        comp_idx = multi_rows[mrow_idx][col_idx][0]
                        comp_height = multi_rows[mrow_idx][col_idx][2]
                        if (comp_idx, comp_height) not in components:
                            components.append((comp_idx, comp_height))
                    except KeyError:
                        components.append((-1, 1000000))
                    except IndexError:
                        components.append((-1, 1000000))
                columns.append(components)
            # Sort the components in the columns by height and remove height variable
            t_columns = []
            for col in columns:
                if len(col) < 2:
                    t_columns.append([col[0][0]])
                    continue
                col.sort(key=lambda y: y[1])
                t_col = []
                for col_comp in col:
                    t_col.append(col_comp[0])
                t_columns.append(t_col)
            columns = t_columns
            
            for col_idx,col in enumerate(columns):
                col_dict = {}
                for c_row_idx,c_row in enumerate(col):
                    multi_row_dict = {}
                    comp_idx = int(c_row)
                    try:
                        comp_label = int(saniticed_bb_results['labels'][comp_idx])
                    except ValueError:
                        comp_label = 13
                    if len(col) == 1:
                        col_dict['component'] = labels[comp_label-1]
                    multi_row_dict['component'] = labels[comp_label-1]

                    try:
                        if len(col) == 1:
                            col_dict['text'] = word_to_idx[comp_idx]
                        multi_row_dict['text'] = word_to_idx[comp_idx]
                    except KeyError:
                        pass

                    for ref in ref_to_obj:
                        if ref[0] == comp_idx:
                            try:
                                if len(col) == 1:
                                    col_dict['reference'] = word_to_idx[ref[1]]
                                multi_row_dict['reference'] = word_to_idx[ref[1]]
                            except KeyError:
                                if debug_print:
                                    print('Error could not resolve a reference value to an object, skipping reference')
                            break;
                    
                    if len(col) > 1:
                        try:
                            col_dict['rows'][str(c_row_idx)] = multi_row_dict
                        except KeyError:
                            col_dict['rows'] = {}
                            col_dict['rows'][str(c_row_idx)] = multi_row_dict

                try:
                    row_dict['cols'][str(col_idx)] = col_dict
                except KeyError:
                    row_dict['cols'] = {}
                    row_dict['cols'][str(col_idx)] = col_dict
        else:
            for col_idx,col in enumerate(row):
                col_dict = {}
                comp_idx = int(col[0])
                try:
                    comp_label = int(saniticed_bb_results['labels'][comp_idx])
                except ValueError:
                    comp_label = 13

                if len(row) == 1:
                    row_dict['component'] = labels[comp_label-1]
                col_dict['component'] = labels[comp_label-1]
                    
                try:
                    if len(row) == 1:
                        row_dict['text'] = word_to_idx[comp_idx]
                    col_dict['text'] = word_to_idx[comp_idx]
                except KeyError:
                    pass
                for ref in ref_to_obj:
                    if ref[0] == comp_idx:
                        try:
                            if len(row) == 1:
                                row_dict['reference'] = word_to_idx[ref[1]]
                            col_dict['reference'] = word_to_idx[ref[1]]
                        except KeyError:
                            if debug_print:
                                print('Error could not resolve a reference value to an object, skipping reference')
                        break;
                if len(row) > 1:
                    try:
                        row_dict['cols'][str(col_idx)] = col_dict
                    except KeyError:
                        row_dict['cols'] = {}
                        row_dict['cols'][str(col_idx)] = col_dict

        json_dict['rows'][str(row_idx)] = row_dict
    
    return json.dumps(json_dict)

# A function for utilizing the definitions file
def get_component_definitions(file_name, labels_to_idx):
    # Gets the component definitions by the following format
    # must_have_text   must_have_reference    can_have_text    can_have_reference    similarity_to_components[component.id]    
    components_definition = []
        
    f = open('./'+file_name, "r")
    json_data = json.load(f)
    for key in json_data['components'].keys():
        if key == "Paragraph":
            continue
        
        data = json_data['components'][key]

        hasText = type(data['hasText']) is not bool
        requiredText = data['hasText'] == 'required'
        
        hasRef = type(data['hasReference']) is not bool
        requiredRef = data['hasReference'] == 'required'
        label_idxs = []
        for d in data['similarity']:
            label_idxs.append(labels_to_idx[d])
           
        components_definition.append((hasText and requiredText, hasRef and requiredRef, hasText or requiredText, hasRef or requiredRef, label_idxs))
    # Since Paragraph is loosely defined we want it to be last at all times:
    data = json_data['components']['Paragraph']

    hasText = type(data['hasText']) is not bool
    requiredText = data['hasText'] == 'required'

    hasRef = type(data['hasReference']) is not bool
    requiredRef = data['hasReference'] == 'required'
    label_idxs = []
    for d in data['similarity']:
        label_idxs.append(labels_to_idx[d])

    components_definition.append((hasText and requiredText, hasRef and requiredRef, hasText or requiredText, hasRef or requiredRef, label_idxs))

    return components_definition

# Calculates the mAP score based on multiple percentages provided
def calc_multi_map(model, data_loader, percentages=[90,75,50], device=torch.device('cpu'), print_res=True):
    model.eval()
    num_classes = model.roi_heads.box_predictor.cls_score.out_features
    metrics = []
    
    for percentage in percentages:
        metric = calc_map(model, data_loader, device, num_classes, IoU=percentage/100.0)
        metrics.append(metric)
        if print_res:
            print()
            print_map_res(metric, percentage, num_classes)
    
    return metrics
      
# Prints a formation of the scores obtained from calculating mAP score
def print_map_res(metric, percentage, num_classes):
    f = open('./datasets/labels.txt', "r")
    data = f.read().split('\n')
    f.close()
    labels = [data[i] for i in range(len(data))]

    print('Values @'+str(percentage)+':')
    print(metric[1])
    print()

    formating = [2,2,2,2,1,1,1,1,1,1,2,1,2]
    for i in range(num_classes-1):
        print(labels[i],'\t'*formating[i]+'->',metric[0][i])
        
# Evaluates the OCR based on the specifications in the thesis
def ocr_eval(data_loader, IoU=0.5, rotations=False, spellcheck=False, filterimg=False): # JUMP_BACK
    reader = easyocr.Reader(['en'], gpu = False)
    
    sharpen_kernel = numpy.array([[-1,-1,-1], [-1,15,-1], [-1,-1,-1]])
    correct_guesses = 0
    guesses = 0   
    characters_correct = 0
    characters_total = 0
    
    epoch_time = time.time()
    batch_nr = 0
    data_loader_size = 0
    
    for images, targets in data_loader:
        if data_loader_size == 0:
            data_loader_size = len(images)
        batch_nr += 1
        for i in range(len(images)):
            guesses += len(targets[i]['boxes'])
            for c in range(len(targets[i]['labels'])):
                characters_total += len(targets[i]['labels'][c])
            t_image = images[i].numpy()[0].transpose(0,1)
            image = numpy.zeros(t_image.shape, dtype=numpy.uint8)
            image[:,:] = t_image[:,:]*255
            postprocessed = False
            if rotations:
                pred = predict_text(image, spellcheck=spellcheck, raw_file=True, filtering=filterimg)
                postprocessed = True
                r_easy_ocr = []
                for c in range(len(pred['boxes'])):
                    r_easy_ocr.append((pred['boxes'][c], pred['words'][c]))
            elif filterimg:
                sharpen = cv2.filter2D(image, -1, sharpen_kernel)
                thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                r_easy_ocr = reader.readtext(thresh)
            else:
                r_easy_ocr = reader.readtext(image)
            
            for pred in r_easy_ocr:
                if postprocessed:
                    box,word = pred
                else:
                    box,word,_ = pred
                    x,y = box[0]
                    x2,y2 = box[2]
                    box = [int(x), int(y), int(x2), int(y2)]
                for c in range(len(targets[i]['boxes'])):
                    target = targets[i]['boxes'][c]
                    iou = calc_iou(target.detach().numpy(), box)
                    if iou > IoU:
                        correct_guesses += 1
                        for g in range(len(targets[i]['labels'][c])):
                            try:
                                letter = targets[i]['labels'][c][g]
                                if letter == '_':
                                    letter = ' '
                                
                                guessed_letter = word[g]

                                if letter == guessed_letter:
                                    characters_correct += 1
                            except IndexError:
                                break;
                        break;
            print(
                '\r[Eval] OCR [{}/{}]\tEpoch time elapsed: {}'.format(
                    (batch_nr-1)*data_loader_size+i, len(data_loader)*data_loader_size, str(datetime.timedelta(seconds=round(time.time()-epoch_time)))
                ),
                end=''
            )
    print()
    print('Evaluation results:\tbox_accuracy = ' + str((correct_guesses/guesses)*100)[:5] + '%\tcharacter_accuracy = ' + str((characters_correct/characters_total)*100)[:5] + '%')

# Evaluates the AI-Engine based on a semantic similarity described in the thesis
def semantic_eval(model, eval_folder, IoU=0.5, disregard_comp=[10,11,12], priority_comp=[11], min_score=[0.80, 0.40]):
    image_folder = eval_folder+'/images/'
    label_folder = eval_folder+'/labels/'
    comp_defs = get_component_definitions('datasets/definitions.json', labels_to_idx)
        
    f = open('./datasets/labels.txt', "r")
    data = f.read().split('\n')
    f.close()
    idx_to_labels = [data[i] for i in range(len(data))]

    scores = []
    
    for image_name in os.listdir(image_folder):
        file_name = image_name[:-4]
        print("Calculating:",file_name)
        f = open(label_folder+file_name+'.txt', 'r')
        data = f.read()
        f.close()
        label_json = json.loads(data)
        
        image = image_folder+image_name
        
        prediction = predict_model(model, image, IoU=IoU, disregard_comp=disregard_comp, priority_comp=priority_comp)
        txt_prediction = predict_text(image)
        predicted_semantic = post_process_results((prediction, txt_prediction), comp_defs, idx_to_labels, min_score=min_score, debug_print=False)
        predicted_json = json.loads(predicted_semantic)
        
        struct, spec = single_semantic_eval(predicted_json, label_json)
        print("Results: Structure =",struct,"Specification =",spec)
        scores.append((struct, spec))
        
    structure_correctness_overall = 0
    specification_correctness_overall = 0
    for (structure, specification) in scores:
        structure_correctness_overall += structure
        specification_correctness_overall += specification
        print(structure, specification)
        
    return structure_correctness_overall/len(scores), specification_correctness_overall/len(scores)
    
# Evaluates a single semantic representation obtained from an image, utilized in semantic_eval
def single_semantic_eval(semantic, true_semantic):
    true_nr_keys = 0
    predicted_nr_keys = 0

    true_nr_specifications = len(true_semantic['objects'].keys())
    predicted_nr_specifications = len(semantic['objects'].keys())

    pred_components, pred_keys = iterate_semantic_row(semantic, [], 0)
    true_components, true_keys = iterate_semantic_row(true_semantic, [], 0)

    true_nr_keys += true_keys
    predicted_nr_keys += pred_keys

    for i in range(len(true_components)):
        true_row_len = len(true_components[i])
        try:
            pred_row_len = len(pred_components[i])
        except IndexError:
            pred_row_len = 0

        correctness = pred_row_len/true_row_len
        if correctness > 1:
            diff_row_len = pred_row_len-true_row_len
            predicted_nr_specifications += pred_row_len-diff_row_len*2
        else:
            predicted_nr_specifications += pred_row_len

        true_nr_specifications += true_row_len

        for c in range(len(true_components[i])):
            t_comp = true_components[i][c]
            try:
                temp_p_comp = pred_components[i]
                p_comp = temp_p_comp[c]
            except IndexError:
                p_comp = {}

            t_attributes = 0
            p_attributes = 0


            if 'text' in t_comp.keys():
                t_attributes += 1
                if 'text' in p_comp.keys():
                    p_attributes += text_similarity(t_comp['text'], p_comp['text'])
            elif 'text' in p_comp.keys():
                p_attributes += 1

            if 'reference' in t_comp.keys():
                t_attributes += 1 
                if 'reference' in p_comp.keys(): 
                    p_attributes += text_similarity(t_comp['reference'], p_comp['reference'])
            elif 'reference' in p_comp.keys(): 
                p_attributes += 1

            if not ('component' in p_comp.keys()):
                t_attributes += 1
            elif t_comp['component'] != p_comp['component']:
                t_attributes += 1

            true_nr_specifications += t_attributes
            predicted_nr_specifications += p_attributes

    structure_correctness = predicted_nr_keys/true_nr_keys
    if structure_correctness > 1:
        structure_correctness = true_nr_keys/predicted_nr_keys

    specification_correctness = predicted_nr_specifications/true_nr_specifications
    if specification_correctness > 1:
        specification_correctness = true_nr_specifications/predicted_nr_specifications

    return structure_correctness,specification_correctness

# Iterates the structure found within the semantic provided
def iterate_semantic_row(semantic, components, nr_keys):  
    nr_keys += len(semantic['rows'])
    for i in range(len(semantic['rows'])):
        row = semantic['rows'][str(i)]
        comp_list = []
        if 'component' in row.keys():
            comp_list.append(row)
        else:
            nr_keys += len(row['cols'].keys())
            
            for c in range(len(row['cols'])):
                col = row['cols'][str(c)]
                if 'component' in col.keys():
                    comp_list.append(col)
                else:
                    components, nr_keys = iterate_semantic_row(col, components, nr_keys)

        components.append(comp_list)    
    return components, nr_keys

# Evaluates the similarity between two texts based on character location
def text_similarity(true_text, pred_text):
    score = 0
    true_text = true_text.lower()
    pred_text = pred_text.lower()
    for i in range(len(true_text)):
        try:
            if true_text[i] == pred_text[i]:
                score += 1
        except IndexError:
            break;
            
    return score/len(true_text)    
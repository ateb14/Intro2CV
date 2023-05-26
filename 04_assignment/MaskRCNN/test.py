from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt




# the outputs includes: 'boxes', 'labels', 'masks', 'scores'

def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):
    classes = [1, 2, 3]
    aps = []

    for cls in classes: # per class app
        # select all the ground truth objects of this class
        obj_num = 0
        for img_idx, gt_labels in enumerate(gt_labels_list):
            for obj_idx, gt_label in enumerate(gt_labels['labels']):
                if gt_label == cls:
                   obj_num += 1
        if obj_num == 0:
            continue

        pred_objs = []
        for img_idx, pred_labels in enumerate(output_list): # different images
            for obj_idx, pred_label in enumerate(pred_labels['labels']): # different objects detected in the same image
                iou = 0
                gt_id = (-1, -1) # (-1, -1) means misclassified
                iou_gate = False # iou_gate = False means bounding box is not correct

                # only consider the prediction of this class
                if pred_label == cls:
                    gt = gt_labels_list[img_idx]
                    if gt['labels'][0] == cls:
                        gt_id = (img_idx, 0)
                        gt_mask = gt['masks'][0].detach().numpy()
                        pred_mask = pred_labels['masks'][obj_idx].detach().numpy().reshape(gt_mask.shape)
                        pred_mask = (pred_mask > 0.5).astype(np.uint8)
                        iou = np.sum(gt_mask & pred_mask) / np.sum(gt_mask | pred_mask)

                        iou_gate = iou > iou_threshold # otherwise, FP

                pred_objs.append({"gt_id": gt_id, 
                                  "confidence": pred_labels['scores'][obj_idx], 
                                  "iou_gate": iou_gate}
                                  )     
                
        # sort pred_objs by confidence
        pred_objs.sort(key=lambda x : x['confidence'], reverse=True)

        # compute TP and FP of different confidence threshold
        TP_recall, TP_precision = 0, 0
        recall_set = set()
        recall_to_precision = {}
        for bound in range(0, len(pred_objs)):
            pred_obj = pred_objs[bound]
            # precision: the ratio of bboxes that satisfy IoU > x% threshold
            if pred_obj['iou_gate']:
                TP_precision += 1
            # recall: the ration of correctly classified objects
            if pred_obj['gt_id'] != (-1, -1):
                if pred_obj['gt_id'] not in recall_set:
                    recall_set.add(pred_obj['gt_id'])
                    TP_recall += 1

            precision = TP_precision / (bound+1)
            recall = TP_recall / obj_num

            if recall not in recall_to_precision or precision > recall_to_precision[recall]:
                recall_to_precision[recall] = precision
        recall_to_precision = sorted(recall_to_precision.items(), key = lambda x : x[0])
        recalls = [x[0] for x in recall_to_precision]
        precisions = [x[1] for x in recall_to_precision]
        # perform interpolation
        x = np.linspace(0, 1, 11)
        y = np.interp(x, recalls, precisions)
        ap = np.mean(y)
        aps.append(ap)
                        
    mAP_segmentation = np.mean(aps)

    return mAP_segmentation



def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    
    classes = [1, 2, 3]
    aps = []

    for cls in classes: # per class app
        # select all the ground truth objects of this class
        obj_num = 0
        for img_idx, gt_labels in enumerate(gt_labels_list):
            for obj_idx, gt_label in enumerate(gt_labels['labels']):
                if gt_label == cls:
                   obj_num += 1
        if obj_num == 0:
            continue

        pred_objs = []
        for img_idx, pred_labels in enumerate(output_list): # different images
            for obj_idx, pred_label in enumerate(pred_labels['labels']): # different objects detected in the same image
                iou = 0
                gt_id = (-1, -1) # (-1, -1) means misclassified
                iou_gate = False # iou_gate = False means bounding box is not correct

                # only consider the prediction of this class
                if pred_label == cls:
                    gt = gt_labels_list[img_idx]
                    if gt['labels'][0] == cls:
                        gt_id = (img_idx, 0)
                        gt_box = gt['boxes'][0]
                        pred_box = pred_labels['boxes'][obj_idx]
                        # compute union area
                        gt_area = gt['area'][0]
                        pred_area = (pred_box[3] - pred_box[1]) * (pred_box[2] - pred_box[0])
                        union_area = gt_area + pred_area
                        # compute intersection area
                        xmin = max(gt_box[0], pred_box[0])
                        ymin = max(gt_box[1], pred_box[1])
                        xmax = min(gt_box[2], pred_box[2])
                        ymax = min(gt_box[3], pred_box[3])
                        intersection_area = max(0, xmax - xmin) * max(0, ymax - ymin)
                        # compute iou
                        iou = intersection_area / (union_area - intersection_area)
                        iou_gate = iou > iou_threshold # otherwise, FP

                pred_objs.append({"gt_id": gt_id, 
                                  "confidence": pred_labels['scores'][obj_idx], 
                                  "iou_gate": iou_gate}
                                  )     
                
        # sort pred_objs by confidence
        pred_objs.sort(key=lambda x : x['confidence'], reverse=True)

        # compute TP and FP of different confidence threshold
        TP_recall, TP_precision = 0, 0
        recall_set = set()
        recall_to_precision = {}
        for bound in range(0, len(pred_objs)):
            pred_obj = pred_objs[bound]
            # precision: the ratio of bboxes that satisfy IoU > x% threshold
            if pred_obj['iou_gate']:
                TP_precision += 1
            # recall: the ration of correctly classified objects
            if pred_obj['gt_id'] != (-1, -1):
                if pred_obj['gt_id'] not in recall_set:
                    recall_set.add(pred_obj['gt_id'])
                    TP_recall += 1

            precision = TP_precision / (bound+1)
            recall = TP_recall / obj_num

            if recall not in recall_to_precision or precision > recall_to_precision[recall]:
                recall_to_precision[recall] = precision
        recall_to_precision = sorted(recall_to_precision.items(), key = lambda x : x[0])
        recalls = [x[0] for x in recall_to_precision]
        precisions = [x[1] for x in recall_to_precision]
        # perform interpolation
        x = np.linspace(0, 1, 11)
        y = np.interp(x, recalls, precisions)
        ap = np.mean(y)
        aps.append(ap)
                        
    mAP_detection = np.mean(aps)
    return mAP_detection







dataset_test = SingleShapeDataset(10)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

num_classes = 4
 
# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')


# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(torch.load(r'/Users/wangyiyang/code/Intro2CV/04_assignment/MaskRCNN/results/maskrcnn_2.pth',map_location='cpu'))

model.eval()
path = "results/" 
# # save visual results
# for i in range(10):
#     imgs, labels = dataset_test[i]
#     output = model([imgs])
    # plot_save_output(path+str(i)+"_result.png", imgs, output[0])

# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(10):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        plot_save_output(path+str(i)+"_result.png", imgs, output[0])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)


np.savetxt(path+"mAP.txt",np.asarray([mAP_detection, mAP_segmentation]))


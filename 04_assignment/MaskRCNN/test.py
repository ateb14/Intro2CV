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




# the outputs includes: 'boxes', 'labels', 'masks', 'scores'

def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):



    return mAP_detection



def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):




    return mAP_segmentation







dataset_test = SingleShapeDataset(10)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

num_classes = 4
 
# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')


# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(torch.load(r'your_weight',map_location='cpu'))



model.eval()
path = "results/" 
# # save visual results
for i in range(10):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path+str(i)+"_result.png", imgs, output[0])


# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(10):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)


np.savetxt(path+"mAP.txt",np.asarray([mAP_detection, mAP_segmentation]))


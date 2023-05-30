import os
from matplotlib import image
import numpy as np
import torch
import torch.utils.data
import random
import cv2
import math
from utils import plot_save_dataset


class SingleShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h//4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        if shape_id == 1:
            cv2.rectangle(mask, (x-s, y-s), (x+s, y+s), 1, -1)
            cv2.rectangle(img, (x-s, y-s), (x+s, y+s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray([random.randint(0, 255)
                                 for _ in range(3)])[None, None, :]

        obj_ids = np.zeros((n_class))

        shape_code = random.randint(1, 3)
        self._draw_shape(img, masks[0, :], shape_code)
        obj_ids[0] = shape_code

        boxes = np.zeros((n_class, 4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0, :] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


# ----------TODO------------
# Implement ShapeDataset.
# Refer to `SingleShapeDataset` for the shape parameters
# ----------TODO------------


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def rand_draw_shape(self, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h//4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        if shape_id == 1:
            cv2.rectangle(mask, (x-s, y-s), (x+s, y+s), 1, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
        return y, x, s, color

    def fix_draw_shape(self, img, shape_id, y, x, s, color):
        if shape_id == 1:
            cv2.rectangle(img, (x-s, y-s), (x+s, y+s), color, -1)

        elif shape_id == 2:
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, idx):
        np.random.seed(idx)
        n_shapes = random.randint(1, 3)

        # prepare the image and masks
        img = np.zeros((self.h, self.w, 3))
        masks = np.zeros((n_shapes, self.h, self.w))
        bg_color = np.asarray([random.randint(0, 255)
                              for _ in range(3)])  # background color
        img[..., :] = bg_color[None, None, :]
        shape_codes = np.random.randint(1, 4, size=n_shapes)  # shapes

        # prepare the bounding boxes and labels
        boxes = np.zeros((n_shapes, 4))
        labels = np.zeros((n_shapes))

        current_mask = np.zeros((self.h, self.w))
        img_shape_info = []

        # shapes with smaller index should be drawn on top of the ones with larger index
        # so the drawing process is defered after the loop
        for i in range(n_shapes):
            while 1:  # keep drawing shapes until overlap is small enough
                masks[i, :] = 0
                y, x, s, color = self.rand_draw_shape(
                    masks[i, :], shape_codes[i])
                if i == 0:
                    # record the shape information for it should be drawn on top of the later ones
                    img_shape_info.append([shape_codes[i], y, x, s, color])
                    # update the current mask to the whole union area
                    current_mask = masks[i, :]
                else:  # check overlap of the current shape with all the previous ones
                    remove_threshold = 0.75
                    current_shape_area = np.sum(masks[i, :])
                    # cut out the intersection area from the original mask
                    masks[i, :] = np.logical_and(
                        masks[i, :], np.logical_not(current_mask))
                    remaining_shape_percentage = np.sum(
                        masks[i, :]) / current_shape_area

                    if remaining_shape_percentage < remove_threshold:  # if the remaining shape is too small, ignore it
                        continue
                    img_shape_info.append([shape_codes[i], y, x, s, color])
                    current_mask = np.logical_or(current_mask, masks[i, :])

                # set the bounding box
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes[i, :] = np.asarray([xmin, ymin, xmax, ymax])
                break

        # draw the shapes on the image in a reverse order
        for shape_info in reversed(img_shape_info):
            self.fix_draw_shape(
                img, shape_info[0], shape_info[1], shape_info[2], shape_info[3], shape_info[4])

        # remove the suppressed shapes
        device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        boxes = torch.as_tensor(boxes, dtype=torch.float32).to(device)
        labels = torch.as_tensor(shape_codes, dtype=torch.int64).to(device)
        masks = torch.as_tensor(masks, dtype=torch.uint8).to(device)
        area = (boxes[:, 3] - boxes[:, 1]) * \
            (boxes[:, 2] - boxes[:, 0]).to(device)

        image_id = torch.tensor([idx]).to(device)
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img).to(device)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


if __name__ == '__main__':
    # dataset = SingleShapeDataset(10)
    dataset = ShapeDataset(10)
    path = "results/"

    for i in range(10):
        print(i)
        imgs, labels = dataset[i]
        print(labels)
        plot_save_dataset(path+str(i)+"_data.png", imgs, labels)

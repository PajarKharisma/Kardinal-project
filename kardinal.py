import torch
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import pickle as pkl
import cv2
import numpy as np
import random
import uuid
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result

import nnArch.darknet as darknet
import nnArch.siamese as siamese
import nnArch.basic_siamese as basic_siamese

class config():
    yolo_cfg_path = 'config/yolov3.cfg'
    yolo_models_path = 'models/yolov3.weights'
    reid_models_path = 'models/re-id-old.pth'
    class_names_path = 'config/coco.names'
    colors_path = 'config/pallete'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False

    reid_thresh = 0.5
    obj_thresh = 0.5
    nms_thresh = 0.4

class PersonId():
    def __init__(self, label='', tensor=None, color=None, bbox=None, frame=1):
        self.label = label
        self.tensor = tensor
        self.color = color
        self.bbox = bbox
        self.frame = frame

    def set_label(self, label):
        self.label = label
    
    def set_tensor(self, tensor):
        self.tensor = tensor

    def set_color(self, color):
        self.color = color

    def set_bbox(self, bbox):
        self.bbox = bbox

    def set_frame(self, frame):
        self.frame = frame

    def get_label(self):
        return self.label
    
    def get_tensor(self):
        return self.tensor

    def get_color(self):
        return self.color

    def get_bbox(self):
        return self.bbox

    def get_frame(self):
        return self.frame

    def get_dist(self, tensor2):
        euclidean_distance = F.pairwise_distance(self.tensor, tensor2)
        d = float(euclidean_distance.item())
        d = abs((1 / (1 + d)) - 1)

        return d

class Kardinal():
    def __init__(self, obj_thresh=config.obj_thresh, nms_thresh=config.nms_thresh):
        self.yolo_model = darknet.Darknet(config.yolo_cfg_path)
        self.yolo_model.load_weights(config.yolo_models_path)
        self.yolo_model.to(config.device)
        self.reid_model = siamese.BstCnn()
        self.reid_model.load_state_dict(torch.load(config.reid_models_path, map_location=config.device))
        self.reid_model.to(config.device)
        self.reid_model.eval()

        self.colors = pkl.load(open(config.colors_path, "rb"))
        self.classes = self.load_classes(config.class_names_path)

        self.input_size = [int(self.yolo_model.net_info['height']), int(self.yolo_model.net_info['width'])]
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh

        self.databases = []

    def load_classes(self, namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def draw_bbox(self, img, bbox, color, label):
        p1 = tuple(bbox[1:3].int())
        p2 = tuple(bbox[3:5].int())

        # kotak orang
        cv2.rectangle(img, p1, p2, color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])

        # kotak text
        cv2.rectangle(img, p3, p4, color, -1)
        cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 1)

    def crop_img(self, img, bboxs):
        imgs = []
        for bbox in bboxs:
            x1, y1 = tuple(bbox[1:3].int())
            x2, y2 = tuple(bbox[3:5].int())

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img_crop = img[y1:y2, x1:x2]
            data = {
                'img' : img_crop,
                'bbox' : bbox
            }
            imgs.append(data)

        return imgs

    def detected(self, img, curr_frame):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensors = cv_image2tensor(img, self.input_size)
        img_tensors = Variable(img_tensors).to(config.device)

        detections = self.yolo_model(img_tensors, config.cuda).cpu()
        detections = process_result(detections, self.obj_thresh, self.nms_thresh)

        if len(detections) > 0:
            detections = transform_result(detections, [img], self.input_size)
            imgs = self.crop_img(img, detections)

            for i, img_crop in enumerate(imgs):
                cv2.imwrite('crop/'+str(uuid.uuid4().hex)+'.jpg', img_crop['img'])
            
            for i, img_crop in enumerate(imgs):
                img_crop['img'] = cv2.resize(img_crop['img'], (64,128))
                tensor_in = cv_image2tensor(img_crop['img'], self.input_size)
                tensor_in = Variable(tensor_in).to(config.device)

                tensor_out = self.reid_model.forward_once(tensor_in).cpu()
                if len(self.databases) < 1:
                    color = random.choice(self.colors)
                    person_id = PersonId(
                        label='Person '+str(i+1),
                        tensor=tensor_out,
                        color=color,
                        bbox=img_crop['bbox'],
                        frame=curr_frame
                    )
                    self.databases.append(person_id)
                else:
                    min_dist = sys.float_info.max
                    sim_person = None
                    for person in self.databases:
                        dist = person.get_dist(tensor_out)
                        if curr_frame != person.get_frame and dist <= config.reid_thresh and dist < min_dist:
                            min_dist = dist
                            sim_person = person
                            # sim_person.set_label(person.get_label())
                            # sim_person.set_color(person.get_color())
                            sim_person.set_bbox(img_crop['bbox'])
                            sim_person.set_tensor(tensor_out)
                            sim_person.set_frame(curr_frame)

                    if sim_person is not None:
                        for person in self.databases:
                            if person.get_label() == sim_person.get_label():
                                person = sim_person
                                break
                    else:
                        color = random.choice(self.colors)
                        new_person = PersonId(
                            label='Person '+str(len(self.databases)+1),
                            tensor=tensor_out,
                            color=color,
                            bbox=img_crop['bbox'],
                            frame=curr_frame
                        )
                        self.databases.append(new_person)

        for person in self.databases:
            self.draw_bbox(img, person.get_bbox() , person.get_color(), person.get_label())

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def yolov3(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensors = cv_image2tensor(img, self.input_size)
        img_tensors = Variable(img_tensors).to(config.device)

        detections = self.yolo_model(img_tensors, config.cuda).cpu()
        detections = process_result(detections, self.obj_thresh, self.nms_thresh)

        if len(detections) > 0:
            detections = transform_result(detections, [img], self.input_size)
            for detection in detections:
                self.draw_bbox(img, detection, (255,255,255), 'orang')
        
        return img

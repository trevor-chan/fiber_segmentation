import numpy as np
import torch
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.instances import Instances

import data

class BrightfieldPredictor:
    def __init__(self, weights_path=None, confidence=0.7):  
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 30000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell)
        #cfg.INPUT.MASK_FORMAT='bitmask'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence   # set the testing threshold for this model
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        if weights_path is not None:
            cfg.MODEL.WEIGHTS = weights_path
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from 
    
        self.cfg = cfg
        
        MetadataCatalog.get('training_dataset').set(thing_classes=['cell'])
        self.metadata = MetadataCatalog.get('training_dataset')
        
        self.prediction_model = DefaultPredictor(self.cfg)
    
    def train(self, dataset_path):
        DatasetCatalog.register('training_dataset', lambda : data.to_coco(dataset_path))
        self.cfg.DATASETS.TRAIN = ('training_dataset',)
        self.cfg.DATASETS.TEST = ()
        
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
    def predict(self, im):
            outputs = self.prediction_model(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata, 
                           scale=3.0, 
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            return v.get_image()[:, :, ::-1]
        
    def predict_large(self, im, stride=256):
        im_height, im_width, _ = im.shape
        all_instances = []
        
        for i in range(0, im_height, stride):
            for j in range(0, im_width, stride):
                sub_img = im[i:i+stride, j:j+stride, :]
                predictions = self.prediction_model(sub_img)
                sub_instances = offset_instances(predictions['instances'], (j, i), (im_height, im_width))
                all_instances.append(sub_instances)

        all_instances = Instances.cat(all_instances)


        v = Visualizer(im[:, :, ::-1],
                       metadata=self.metadata, 
                       scale=8.0, 
                       #instance_mode=ColorMode.IMAGE_BW
        )
        v = v.draw_instance_predictions(all_instances.to("cpu"))
        return v.get_image()[:, :, ::-1]
    
    
def offset_boxes(boxes, offset):
    new_boxes = boxes.clone()
    i, j = offset
    for box in new_boxes:
        box[0] += i
        box[2] += i
        box[1] += j
        box[3] += j
    return new_boxes


def offset_masks(masks, offset):
    i, j = offset
    polygon_masks = []
    masks = masks.cpu()
    for mask in masks:
        polygon_mask = mask_to_polygons(mask)[0]
        for sub_polygon_mask in polygon_mask:
            sub_polygon_mask[::2] += i
            sub_polygon_mask[1::2] += j
        #polygon_mask[0][::2] += i
        #polygon_mask[0][1::2] += j
        polygon_masks.append(polygon_mask)
        
    return polygon_masks

def offset_instances(instances, offset, im_size):
    instance_dict = {
        'pred_boxes': offset_boxes(instances.pred_boxes, offset),
        'scores': instances.scores,
        'pred_classes': instances.pred_classes,
        'pred_masks': offset_masks(instances.pred_masks, offset)
    }
    return Instances(im_size, **instance_dict)

def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]
    return res, has_holes
import pycocotools
import numpy as np
import os
import pickle
from detectron2.structures import BoxMode

def absolute_paths(directory):
    filenames = sorted(os.listdir(directory))
    return [os.path.join(directory, filename) for filename in filenames]

def compute_bbox(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]

def to_coco(dataset_path):
    image_paths = absolute_paths(os.path.join(dataset_path, 'images'))
    target_paths = absolute_paths(os.path.join(dataset_path, 'targets'))
    
    dataset_dicts = []
    
    for idx, (image_path, target_path) in enumerate(zip(image_paths, target_paths)):
        with open(target_path, 'rb') as f:
            target = pickle.load(f)
        
        record = {}
        record['file_name'] = image_path
        record['image_id'] = idx
        record['height'] = target['size'][1]
        record['width'] = target['size'][0]
        
        objs = []
        for m in target['masks']:
            annotation = {'segmentation': pycocotools.mask.encode(np.asarray(m, order="F")),
                          'bbox': compute_bbox(m),
                          'bbox_mode': BoxMode.XYXY_ABS,
                          'category_id': 0,}
            objs.append(annotation)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts
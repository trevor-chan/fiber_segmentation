from modules import BrightfieldPredictor
import cv2
import os
import matplotlib.pyplot as plt
from sys import argv
import pickle

model = BrightfieldPredictor(weights_path='./models/bright-field.pth', confidence=0.6)

assert len(argv) > 1, "missing data file"

file_name = argv[1]

image = cv2.imread(file_name)

instances = model.predict_large(image)
#instances = instances.to('cpu')
instance_dict = {
    "pred_boxes":instances.pred_boxes.tensor,
    "pred_masks":instances.pred_masks,
    "scores":instances.scores,
    "classes":instances.pred_classes,
    "image_size": instances.image_size,
}

with open(file_name[0:-5]+'_instances.data', 'wb') as filehandle:
    pickle.dump(instance_dict, filehandle)

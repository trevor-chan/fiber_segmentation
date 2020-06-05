from skimage import data, color, io, img_as_float
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def predict_masks(model, image):
    image = (torch.tensor(np.array(image))/255.).permute(2, 0, 1)#.cuda()
    model.eval()
    with torch.no_grad():
        prediction = model([image])
    masks = [mask.cpu().numpy() for mask in prediction[0]['masks']]
    scores = prediction[0]['scores']
    return masks, scores

def composite_masks(image, masks):
    masks = [mask > 0.5 for mask in masks]
    masks = [(np.array(mask).squeeze() * 255).astype(np.uint8) for mask in masks]
    num_objects = len(masks)
    
    for m in range(num_objects):
        mask = Image.fromarray(masks[m])
        
        color = (np.ones_like(image)*np.random.uniform(size=3)*255).astype(np.uint8)
        color = Image.fromarray(color, 'RGB')
        
        image = Image.composite(color, image, mask)
    return image
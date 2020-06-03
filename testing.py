from skimage import data, color, io, img_as_float
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def predict(model, image):
    model.eval()
    with torch.no_grad():
        prediction = model([image.cuda()])
        
    input_image = image.mul(255).permute(1, 2, 0).byte().numpy()
    masks = []
    
    #img_hsv = color.rgb2hsv(image.permute(1, 2, 0))
    
    #alpha = 0.5
    masked_image = Image.fromarray(input_image)
    
    num_objects = prediction[0]['labels'].shape[0]
    
    for m in range(num_objects):
        if prediction[0]['scores'][m] < 0.5:
            continue
        
        mask = Image.fromarray(prediction[0]['masks'][m, 0].mul(255).byte().cpu().numpy(), )
        
        #print(input_image.shape)
        color = (np.ones_like(input_image)*np.random.uniform(size=3)*255).astype(np.uint8)
        color = Image.fromarray(color, 'RGB')
        
        masked_image = Image.composite(color, masked_image, mask)
        
        #mask_color = np.dstack((mask, mask, mask)) * np.array([1, 0, 0])#* #np.random.uniform(3)
        #plt.imshow(mask_color)
        #plt.show()
        #mask_hsv = color.rgb2hsv(mask_color)
        #print(mask_hsv.shape)
        
        #img_hsv[..., 0] = mask_hsv[..., 0]
        #img_hsv[..., 1] = mask_hsv[..., 1] * alpha
    
    #img_masked = color.hsv2rgb(img_hsv)
    return masked_image
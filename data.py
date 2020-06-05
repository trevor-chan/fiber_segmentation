import json
import torch
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import random
from multiprocessing.pool import ThreadPool
import PIL
import pickle

def collate_fn(batch):
    return tuple(zip(*batch))

def get_image_from_url(url):
    while True:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            break
    response.raw.decode_content = True
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        super().__init__()
        
        with open(json_path) as f:
            self.data = json.load(f)
            
    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump({'images': self.images, 'masks': self.masks}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self, load_path):
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.images = data['images']
            self.masks = data['masks']
            
    def transform(self, image, masks):
        image = TF.to_pil_image(image)
        masks = [TF.to_pil_image(mask) for mask in masks]
        """
        # Brightness
        brightness_factor = np.random.normal()*0.2 + 1
        image = TF.adjust_brightness(image, brightness_factor)
        
        # Contrast
        contrast_factor = np.random.normal()*0.2 + 1
        image = TF.adjust_contrast(image, contrast_factor)

        angle = np.random.uniform(-180, 180)
        shear = np.random.normal()*20
        scale = np.random.uniform(0.5, 2.0)
        translate = np.random.randint(-30, 30, size=2).tolist()
        image = TF.affine(image, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=None)
        masks = [TF.affine(mask, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=None) for mask in masks]
        """
        # Random crop
        #i, j, h, w = transforms.RandomCrop.get_params(
        #    image, output_size=(256, 256))
        i=400
        j=100
        h=256
        w=256
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        """
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]
    
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(mask) for mask in masks]
        """
        # Transform to tensor
        image = TF.to_tensor(image)
        masks = [TF.to_tensor(mask) for mask in masks]
        return image, masks
    
    
    def fetch(self, num_threads=16):
        self.images = []
        self.masks = []
        for idx in range(len(self)):
            img_url = self.data[idx]['Labeled Data']
            mask_urls = [obj['instanceURI'] for obj in self.data[idx]['Label']['objects']]

            img = get_image_from_url(img_url)
            masks = list(ThreadPool(num_threads).imap_unordered(get_image_from_url, mask_urls))
            self.images.append(img)
            self.masks.append(masks)
    
        
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        img = self.images[idx]
        masks = self.masks[idx]
        
        img = np.array(img)
        masks = [np.array(mask)[:, :, 0:1] for mask in masks]
        
        img, masks = self.transform(img, masks)
        #return img, masks
        
        masks.insert(0, torch.ones(1, img.shape[1], img.shape[2])) # add background
        
        filtered_masks = []
        filtered_boxes = []
        for mask in masks:
            mask = mask.bool().numpy().squeeze()
            if not mask.any():
                continue
            
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            if ymax - ymin < 2 or xmax - xmin < 2:
                continue
            filtered_masks.append(mask)
            filtered_boxes.append([xmin, ymin, xmax, ymax])
        
        masks = filtered_masks
        boxes = filtered_boxes
        num_objs = len(masks)
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        labels[0] = 0
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target
import time
import requests
from PIL import Image
import PIL
import json
from multiprocessing.pool import ThreadPool
from io import BytesIO

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import pickle
import numpy as np
import os
import shutil
from tqdm import tqdm
from multiprocessing import Process, Queue

def save_mask_target(image, masks, name, dataset_path = 'dataset'):
    image.save(os.path.join(dataset_path, 'images', name + '.png'))
    with open(os.path.join(dataset_path, 'targets', name + '.pkl'), 'wb') as f:
        pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_image_from_url(url):
    while True:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            break
        time.sleep(0.5)
    response.raw.decode_content = True
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def augment(image, masks):
        # Brightness
        brightness_factor = np.random.normal()*0.2 + 1
        image = TF.adjust_brightness(image, brightness_factor)
        
        # Contrast
        contrast_factor = np.random.normal()*0.2 + 1
        image = TF.adjust_contrast(image, contrast_factor)

        # Affine
        angle = np.random.uniform(-180, 180)
        shear = np.random.normal()*20
        scale = np.random.uniform(0.5, 2.0)
        translate = np.random.randint(-30, 30, size=2).tolist()
        image = TF.affine(image, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=None)
        masks = [TF.affine(mask, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=None) for mask in masks]
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 256))
        
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        # Random horizontal flipping
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]
    
        # Random vertical flipping
        if np.random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(mask) for mask in masks]
        # Transform to tensor
        #image = TF.to_tensor(image)
        #masks = [TF.to_tensor(mask) for mask in masks]
        
        # squeeze and binarize
        masks = [(np.array(mask)[:, :, 0] > 0.5).astype(np.uint8) for mask in masks]
        
        # prune masks that have no object or only a sliver of an object
        masks = [mask for mask in masks if mask[10:-10, 10:-10].any()]
        return image, masks

class Worker(Process):
    def __init__(self, task_queue, result_queue, img, masks, out_path):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.img = img
        self.masks = masks
        self.out_path = out_path
        
    def run(self):
        proc_name = self.name
        while True:
            index = self.task_queue.get()
            if index == -1: break
            sub_img, sub_masks = augment(self.img, self.masks)
            target = {'masks': sub_masks, 'size': sub_img.size}
            save_mask_target(sub_img, target, f'{index:05d}', dataset_path=self.out_path)            
            self.result_queue.put(index)
        return
    
def download_dataset(json_path, out_path, samples_per_img=100, num_threads=16, num_processes=4, selected_ids=None):

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(os.path.join(out_path, 'images'))
    os.makedirs(os.path.join(out_path, 'targets'))
    
    
    total_images = 0
    with open(json_path) as f:
        data = json.load(f)
    
    if selected_ids is not None:
        # Filter only selected images
        data = [img_obj for img_obj in data if img_obj['External ID'] in selected_ids]

    task_queue = Queue()
    result_queue = Queue()
                                
    
    with tqdm(total=len(data)*samples_per_img) as pbar:
        for img_obj in data:
            img_url = img_obj['Labeled Data']
            mask_urls = [instance['instanceURI'] for instance in img_obj['Label']['objects']]

            img = get_image_from_url(img_url)
            masks = list(ThreadPool(num_threads).imap_unordered(get_image_from_url, mask_urls))
            
            workers = []
            for proc_index in range(num_processes):
                p = Worker(task_queue, result_queue, img, masks, out_path)
                p.daemon = True
                p.start()
                workers.append(p)
            
            for _ in range(samples_per_img):
                task_queue.put(total_images)
                total_images += 1
                            
            for index in range(samples_per_img):
                i = result_queue.get()
                pbar.update(1)
                
            for index in range(num_processes):
                task_queue.put(-1)
            for worker in workers:
                worker.join()
                
                
                
def main():
    json_path = 'datasets/export-2020-06-04T19_14_29.068Z.json'
    samples_per_img = 100
    
    download_dataset(json_path,
                     'cells_train',
                     samples_per_img=samples_per_img,
                     selected_ids=['image_part_003.jpg'])
    download_dataset(json_path,
                 'cells_val',
                 samples_per_img=samples_per_img,
                 selected_ids=['image_part_006.jpg'])
                
if __name__ == '__main__':
    main()
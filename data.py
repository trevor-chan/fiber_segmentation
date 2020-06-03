import json
import torch
from PIL import Image
import requests
from io import BytesIO

def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        super().__init__()
        
        with open(json_path) as f:
            self.data = json.load(f)
            
        self.transforms = None
        
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        img_url = self.data[idx]['Labeled Data']
        mask_urls = [obj['instanceURI'] for obj in self.data[idx]['Label']['objects']]
        
        img = get_image_from_url(img_url).convert("RGB")
        masks = [get_image_from_url(mask_url) for mask_url in mask_urls]
        
        img = np.array(img)
        masks = [np.array(mask)[:, :, 0] for mask in masks]
        
        num_objs = len(masks)
        # get bounding box coordinates for each mask
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
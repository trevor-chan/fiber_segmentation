import os
import sys

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

class BrightfieldPredictor:
    def __init__(self, weights_path='output/model_final.pth', confidence=0.7):
        MetadataCatalog.get('cells_train').set(thing_classes=['cell'])
        self.metadata = MetadataCatalog.get('cells_train')

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell)
        cfg.MODEL.WEIGHTS = os.path.join(weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence   # set the testing threshold for this model
        cfg.TEST.DETECTIONS_PER_IMAGE = 500

        self.model = DefaultPredictor(cfg)
    
    def predict(self, im):
            outputs = self.model(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata, 
                           scale=3.0, 
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            return v.get_image()[:, :, ::-1]

def main():
    args = sys.argv
    img_url = args[-1]
    
    im = cv2.imread(img_url)
    predictions = self.model(im)
    
    
if __name__ == '__main__':
    main()
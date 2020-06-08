import argparse
from modules import BrightfieldPredictor
import cv2
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path',
                        '-i',
                        default='in.jpeg')
    parser.add_argument('--out_path',
                        '-o',
                        default='out.jpeg')
    parser.add_argument('--confidence',
                        '-c',
                        type=float,
                        default=0.6)
    parser.add_argument('--weights_path',
                        '-w',
                        default='models/256/model_final.pth')
    
    args = parser.parse_args()
    
    image = cv2.imread(args.in_path)
    
    model = BrightfieldPredictor(weights_path=args.weights_path,
                                 confidence=args.confidence)
    
    out_image = model.predict_large(image)
    out_image = Image.fromarray(out_image)
    out_image.save(args.out_path)
    
if __name__ == '__main__':
    main()
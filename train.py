import argparse
from torch.optim.lr_scheduler import StepLR

from modules import *
from data import *
from training import *
from testing import *

################### HYPERPARAMS ###################

BATCH_SIZE = 4
LR = 5e-5
LR_STEP_SIZE = 5
LR_GAMMA = 0.1
EPOCHS = 2000


###################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload_dataset',
                        action='store_true')
    parser.add_argument('--reinitialize_model',
                        action='store_true')
    
    args = parser.parse_args()
    
    d = CellDataset('datasets/export-2020-06-04T19_14_29.068Z.json')
    
    if args.reload_dataset:
        print("Reloading Dataset")
        # SELECT ONLY FULLY LABELED
        """
        selected_ids = ['MC171178.JPG',
                        'MC171181.JPG',
                        'MC171179.JPG',
                        'MC171177.JPG',
                        'MC171180.JPG',
                        'image_part_003.jpg',
                        'image_part_006.jpg']
        """
        selected_ids = ['image_part_003.jpg']
        d.data = [img for img in d.data if img['External ID'] in selected_ids]
        d.fetch()
        #print("Saving Dataset")
        #d.save('datasets/saved_dataset.pth')
    else:
        d.load('dataset/saved_dataset.pth')

    dataloader = torch.utils.data.DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, collate_fn=collate_fn)

    model = get_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    losses = []
    
    if not args.reinitialize_model:
        checkpoint = torch.load('models/model.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losses = checkpoint['losses']

    train(model, optimizer, dataloader, EPOCHS, losses, scheduler)
    
if __name__ == '__main__':
    main()
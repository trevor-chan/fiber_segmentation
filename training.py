import torch

device = 'cuda'

def train(model, optimizer, dataloader, epochs, losses):
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"Epoch \t {epoch}")
        for i, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())
            print(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if i % 20 == 0:print(loss.item())
        if epoch % 5 == 0: torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}, 'models/model.pth')
        
        

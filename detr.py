import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from trail import train_ibll, test_ibll

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __getitem__(self, idx):
        image = self.images[idx]['img']  # Assuming the 'img' item is a numpy array
        image = image.transpose((2, 0, 1))  # Change (H, W, C) to (C, H, W)
        image = image / 255.0  # Normalize to [0, 1]
        image = torch.from_numpy(image).float()  # Convert to tensor

        # Now, convert the target into the correct format
        boxes = self.targets[idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all objects belong to a single class
        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.images)
    
train_targets = [data['bboxes'] for data in train_ibll.data]

if __name__ == "__main__":
    dataset = CustomDataset(train_ibll.data, train_targets)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 40

    for epoch in range(num_epochs):
        model.train()
        i = 0    
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            if i % 50 == 0:
                
                print(f"Iteration.................. #{i} loss: ..............{losses.item()}")

        # Save the model checkpoint
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"modelfront_epoch_{epoch}.pth")

        print(f"Training...............................GPU engaged")
        print(f"[......................................................]") 
        print(f"[......................................................]") 
        print(f"[......................................................]")    
        print(f"Epoch #.............{epoch} loss: .....................{losses.item()}")

import os
import torch
import argparse
import numpy as np
from torch import nn, optim
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="train_crop_datasets")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epoch", type=float, default=30)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--save_path", type=str, default="./checkpoint")
args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)


# get dataset / dataloader
train_transformer = transforms.Compose([np.float32,
                                        transforms.ToTensor(),
                                        fixed_image_standardization])

train_datasets = datasets.ImageFolder(args.train_data,
                                      transform=train_transformer)

train_loader = torch.utils.data.DataLoader(train_datasets,
                                           batch_size=args.batch_size)

classes = list(train_datasets.classes)

model = InceptionResnetV1(pretrained="vggface2",
                          classify=True,
                          num_classes=len(classes)).to(args.device)
# cost
criterion = nn.CrossEntropyLoss().to(args.device)

train_iter = len(train_loader)

for p in model.parameters():
    p.requires_grad = False

for p in model.logits.parameters():
    p.requires_grad = True

# optimizer/scheduler
# optimizer = optim.SGD(model.logits.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[10, 20])

best_acc = 0

# train
for e in range(args.epoch):
    scheduler.step()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for (images, targets) in tqdm(train_loader, total=train_iter):
        images = images.to(args.device)
        targets = targets.to(args.device)

        optimizer.zero_grad()
        # forward
        output = model(images)
        # acc
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        temp_labels = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(temp_labels).float()

        top1_meter.update(correct[:, :1].sum())
        top5_meter.update(correct[:, :5].sum())

        # loss
        loss = criterion(output, targets)
        loss_meter.update(loss.item())
        # backward
        loss.backward()
        # weight update
        optimizer.step()

    train_loss = loss_meter.avg / args.batch_size
    top1_acc = top1_meter.avg / args.batch_size
    top5_acc = top5_meter.avg / args.batch_size

    print(f"Epoch [ {args.epoch} / {e} ] \n"
          f" + TRAIN [ Loss / Top1 Acc / Top5 Acc ] : [ {train_loss} / {top1_acc} / {top5_acc} ]")

    if best_acc < top1_acc:
        torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torchvision
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
import random
import argparse
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from cityscapes import CityScapesDataset
import json
import time
import pandas as pd
import gc

model = deeplabv3_resnet101(weight=DeepLabV3_ResNet101_Weights,num_classes = 13)

parser = argparse.ArgumentParser(description='DeepLab Segmentation')
parser.add_argument('--lr', type = float, default=0.001, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

# function to colourize the bitmask
def colorize_bitmask(bitmask, color_dict):
    height, width = bitmask.shape
    colorized_image = np.zeros((height, width, 3)) # create a 0 array that is of the shape of the bitmask

    for key, color in color_dict.items():
        mask = (bitmask == key)
        colorized_image[mask]= color # assign the color based on the key
    return colorized_image # return colorized image

def collate(batch):
    # Unpack the batch into separate lists for images and masks
    images, masks = zip(*batch)
    
    # Convert the list of images into a batch tensor
    images = torch.stack(images, dim=0)
    
    # Convert the list of masks into a batch tensor
    masks = torch.stack(masks, dim=0)
    
    return images, masks

def compute_accuracy(predicted, targets):
    predicted = predicted.to('cpu')
    targets = targets.to('cpu')

    # Count correct predictions
    correct_predictions = torch.eq(predicted, targets).sum().item()
   
    # Total number of predictions
    total_predictions = targets.numel()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_miou(gt_masks, pred_masks, num_classes):
    gt_masks = gt_masks.cpu()
    pred_masks = pred_masks.detach().cpu()
    intersection = torch.logical_and(gt_masks, pred_masks).float().sum((1, 2))
    union = torch.logical_or(gt_masks, pred_masks).float().sum((1,2))
    if union.sum() == 0:
        iou = 0
    else: 
        iou = intersection / union
    miou = torch.mean(iou)
    return miou

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) 
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def validate(model, dataloader,loss_fn):
    valloss = 0
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            output = model(imgs)['out']
            del imgs
            loss = loss_fn(output,masks)
            valloss += loss.item()
            output = torch.nn.functional.softmax(output, dim=1)
            output = torch.argmax(output,dim=1)
            masks = torch.argmax(masks, dim = 1)
            all_labels.extend(masks.cpu().numpy())
            all_preds.extend(output.cpu().numpy())

    return valloss, scores(all_labels,all_preds,13)

def train(model,dataloader, optimizer,sched,loss_fn):
    train_loss = 0
    all_labels = []
    all_preds = []

    for imgs,masks in dataloader:
        optimizer.zero_grad()
        imgs = imgs.to(device)
        masks = masks.to(device)
        model.train()
        output = model(imgs)['out']
        loss = loss_fn(output,masks)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        sched.step()

        output = torch.nn.functional.softmax(output, dim=1)
        output = output.detach().cpu()
        masks = masks.detach().cpu()
        
        output = torch.argmax(output,dim=1)
        masks = torch.argmax(masks, dim = 1)

        all_labels.extend(masks.cpu().numpy())
        all_preds.extend(output.cpu().numpy())
    return train_loss, scores(all_labels,all_preds,13)


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    best_loss = float('inf')
    df = pd.DataFrame(columns=["Epoch", "Train_Loss", "Train_Pixel_Acc", "Train_Mean_Acc","Train_FW_IOU","Train_Mean_IoU",
                               "Val_Loss", "Val_Pixel_Acc", "Val_Mean_Acc","Val_FW_IOU","Val_Mean_IoU"])

    color_dict = {0: (0.502, 0.251, 0.502),
    1: (0.957, 0.137, 0.910),
    2: (0.275, 0.275, 0.275),
    3: (0.275, 0.510, 0.706),
    4: (0.863, 0.078, 0.235),
    5: (1.0, 0.0, 0.0),
    6: (0.0, 0.0, 0.557),
    7: (0.0, 0.0, 0.275),
    8: (0.0, 0.235, 0.392),
    9: (0.0, 0.314, 0.392),
    10: (0.0, 0.0, 0.920),
    11: (0.467, 0.043, 0.125),
    12: (0, 0, 0)}


    eqv_dict = {7: 0,       #sets mask_ids to their approriate train_ids that will be evaluated after prediction 
                8: 1,
                11: 2,
                12: 2,
                13: 2,
                23: 3,
                24: 4,
                25: 5,
                26: 6,
                27: 7,
                28: 8,
                31: 9,
                32: 10,
                33: 11,
                255: 12}


    # Initialize a list containing all the object classes to identify the class from the index
    object_class = ["road",
            "sidewalk",
            "background",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "unknown"]

    ignoreeval_list = np.array([0, 4, 5, 6, 9, 10, 14, 
                                15, 16, 17, 18, 19, 20, 21, 22, 29, 30,34]) #mask_ids in the cityscapes bitmasks that are not evaluated


    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((256,512))  # Resize the input image to the expected size
            # Convert the image to a PyTorch tensor
    ])

    data = CityScapesDataset('/home/harishs/projects/def-akilan/harishs/objdet/gtFine/train',
                            '/home/harishs/projects/def-akilan/harishs/objdet/leftImg8bit/train',
                            ignoreeval_list=ignoreeval_list,
                            eqv_dict=eqv_dict,
                            transform=transform)
    
    valdata = CityScapesDataset('/home/harishs/projects/def-akilan/harishs/objdet/gtFine/val',
                        '/home/harishs/projects/def-akilan/harishs/objdet/leftImg8bit/val',
                        ignoreeval_list=ignoreeval_list,
                        eqv_dict=eqv_dict,
                        transform=transform)
    

    train_loader = DataLoader(data, batch_size=args.batch_size,collate_fn=collate,shuffle=True)
    val_loader = DataLoader(valdata, batch_size=args.batch_size,collate_fn=collate,shuffle=True)

    max_lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(),lr = max_lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=args.max_epochs,steps_per_epoch=len(train_loader))

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    loss_list = []
    val_loss_list = []
    acc_list = [] 
    val_acc_list = []
    miou_list = []
    val_miou_list =[]

    best_loss = float('inf')

    for epoch in range(args.max_epochs):
        start = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        train_loss, train_scores = train(model,train_loader,optimizer,sched,loss_fn)
        val_loss, val_scores  = validate(model,val_loader,loss_fn)

        with open('/home/harishs/projects/def-akilan/harishs/objdet/deeplab101_checkpoints/deeplab101train.json', 'a') as f:
            json.dump(train_scores, f)
            f.write('\n') 

        with open('/home/harishs/projects/def-akilan/harishs/objdet/deeplab101_checkpoints/deeplab101val.json', 'a') as f:
            json.dump(val_scores, f)
            f.write('\n') 

        df = df._append({"Epoch": epoch + 1, "Train_Loss": train_loss, "Train_Pixel_Acc": train_scores['Pixel Accuracy'], "Train_Mean_Acc": train_scores['Mean Accuracy'],
                         "Train_FW_IOU": train_scores['Frequency Weighted IoU'],"Train_Mean_IoU":train_scores["Mean IoU"],"Val_Loss":val_loss, 
                         "Val_Pixel_Acc":val_scores["Pixel Accuracy"], "Val_Mean_Acc": val_scores["Mean Accuracy"],"Val_FW_IOU" : val_scores["Frequency Weighted IoU"],
                         "Val_Mean_IoU":val_scores['Mean IoU']}, ignore_index=True)
        
        print(f'Epoch Train [{epoch+1}/{args.max_epochs}], Loss: {train_loss/len(train_loader)}, Accuracy:{train_scores["Pixel Accuracy"]*100}, mIoU:{train_scores["Mean IoU"]*100}',flush=True)
        print(f'Epoch Validate [{epoch+1}/{args.max_epochs}], Loss: {val_loss/len(train_loader)}, Accuracy:{val_scores["Pixel Accuracy"]*100}, mIoU:{val_scores["Mean IoU"]*100}',flush=True)
        print(f'Time taken: {(time.time()-start)/60}',flush = True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': sched.state_dict(),
                'best_loss': best_loss
            },f'/home/harishs/projects/def-akilan/harishs/objdet/deeplab101_checkpoints/{epoch+1}_checkpoint.pth')
            print('Model Saved \n')

    df.to_csv('deeplab101_metrics.csv',index = False)

    torch.save(model.state_dict(),'/home/harishs/projects/def-akilan/harishs/objdet/deeplab101_final_model.pth')

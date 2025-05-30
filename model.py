import torch 
import torch.nn as nn
import os
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from kittidataloader import CustomImageDataset
from collections import OrderedDict
import argparse
from metrics import compute_map
import gc
from torchvision.models.segmentation import deeplabv3_resnet101
from lightning.pytorch.callbacks import TQDMProgressBar,ModelCheckpoint,EarlyStopping
import lightning as pl
import pandas as pd
import warnings
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Faster RCNN, distributed data parallel test')
parser.add_argument('--lr', type = float, default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--name', type=str, default=4, help='')

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return psi*x
    
class combineModel(nn.Module):
    # g is skip, x is previous; g = semseg x = fasterrcnn
    def __init__(self,seg_model,faster_model):
        super(combineModel, self).__init__()
        self.seg_model = seg_model
        self.target_layers = ['backbone.layer1','backbone.layer2','backbone.layer3','backbone.layer4']

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.transform = faster_model.transform
        self.pre = list(faster_model.backbone.body.children()) 
        self.pre = nn.Sequential(*self.pre[0:4])
        self.layer1 = faster_model.backbone.body.layer1
        self.layer2 = faster_model.backbone.body.layer2
        self.layer3 = faster_model.backbone.body.layer3
        self.layer4 = faster_model.backbone.body.layer4
        self.fpn = faster_model.backbone.fpn
        self.rpn = faster_model.rpn
        self.roi = faster_model.roi_heads
        # self.attn1 = AttentionBlock(f_g = 256 ,f_l = 256, f_int = 256)
        self.attn2 = AttentionBlock(f_g = 512 ,f_l = 512, f_int = 512)
        self.attn3 = AttentionBlock(f_g = 1024 ,f_l = 1024, f_int = 1024)
        self.attn4 = AttentionBlock(f_g = 2048 ,f_l = 2048, f_int = 2048)

        for name, module in self.seg_model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(lambda module, input, output, name=name: self.hook(module, input, output, name))
        
    def hook(self,module,input,output,name):
        self.intermediate_output[name] = output

    def forward(self, imgs, target = None):
        
        self.intermediate_output = {}
        
        original_image_shapes = [(img.shape[1], img.shape[2]) for img in imgs]

        imgs_list, target = self.transform(imgs,target)

        self.seg_model.eval()
        with torch.no_grad():
            _ = self.seg_model(imgs_list.tensors)['out']

        output_1 = self.intermediate_output['backbone.layer1']
        output_2 = self.intermediate_output['backbone.layer2']
        output_3 = self.intermediate_output['backbone.layer3']
        # output_3 = F.pad(output_3, (0, 0, 0, 1))  # Pad height dimension
        output_3 = self.max_pool(output_3)
        output_4 = self.intermediate_output['backbone.layer4']
        # output_4 = F.pad(output_4, (0, 0, 0, 1))  # Pad height dimension
        output_4 = self.max_pool(output_4)
        output_4 = self.max_pool(output_4)

        x = self.pre(imgs_list.tensors)
        x = self.layer1(x)
        # x = self.attn1(output_1,x)
        ordered_dict = OrderedDict()
        ordered_dict['0'] = x

        x = self.layer2(x)
        x = self.attn2(output_2,x)
        ordered_dict['1'] = x

        x = self.layer3(x)
        x = self.attn3(output_3,x)
        ordered_dict['2'] = x

        x = self.layer4(x)
        x = self.attn4(output_4,x)
        ordered_dict['3'] = x


        x = self.fpn(ordered_dict)

        if self.training:
            proposals,proposal_losses = self.rpn(imgs_list,x,target)
            detections, detector_losses = self.roi(x,proposals,image_shapes = imgs_list.image_sizes, targets = target)
            detections = self.transform.postprocess(detections, imgs_list.image_sizes, original_image_shapes)
            loss = detector_losses
            loss.update(proposal_losses)
            return loss
        else:
            proposals,proposal_losses = self.rpn(imgs_list,x)
            detections, detector_losses = self.roi(x,proposals,image_shapes = imgs_list.image_sizes)
            detections = self.transform.postprocess(detections, imgs_list.image_sizes, original_image_shapes)
            return detections
        
def main():
    print('Starting....')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate(data):
        # Separate images and targets
        imgs = [item[0] for item in data]
        targets = [{'boxes': item[1]['boxes'],
                    'labels': item[1]['labels']} for item in data]

        # Move images and targets to the device
        imgs = torch.stack(imgs).to(device)
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)

        return imgs, targets
    
    class DeepLabNet(pl.LightningModule):
        def __init__(self,model,steps,val_steps):
            super(DeepLabNet, self).__init__()
            # Load the pre-trained DeepLab model
            self.model = model
            # Initialize lists to store metrics
            self.steps = steps
            self.val_steps = val_steps
            self.train_loss_epoch = 0.0
            self.val_loss_epoch = 0.0

            self.mAP_list = []
            self.accuracy = []
            self.batch_mAP_list = []
            self.batch_accuracy = []
            self.val_batch_mAP_list = []
            self.val_batch_accuracy = []

            self.df = pd.DataFrame(columns=["Epoch", "Train_Loss", "Train_Accuracy", "Train_mAP", "Val_Loss", "Val_Accuracy", "Val_mAP"])
        
        def forward(self, x, tar = None):
            if self.model.training:
                x = self.model(x,tar)
            else:
                x = self.model(x)
            return x
        
        def training_step(self, batch, batch_idx):
            self.model.train()
            x, y = batch
            loss_dict = self(x,y)
            loss = sum(v for v in loss_dict.values())
            loss = loss.mean()
            # Accumulate metrics for the epoch
            self.train_loss_epoch += loss.item()

            self.model.eval()
            with torch.no_grad():
                pred = self(x)
            for i in range(len(x)):
                bbox_pred, labels_pred, scores_pred = pred[i]['boxes'].cpu().numpy(), pred[i]['labels'].cpu().numpy(), pred[i]['scores'].cpu().numpy()
                bbox_gt, labels_gt = y[i]['boxes'].cpu().numpy(), y[i]['labels'].cpu().numpy()

                acc, mAP = compute_map(bbox_pred, labels_pred,scores_pred,bbox_gt,labels_gt,9)
                self.mAP_list.append(mAP)
                self.accuracy.append(acc)
            self.batch_mAP_list.append(np.array(self.mAP_list).mean())
            self.batch_accuracy.append(np.array(self.accuracy).mean())
            self.mAP_list = []
            self.accuracy = []
            return loss

        # def on_train_epoch_end(self):
        #     # Compute average metrics for the epoch
        #     avg_train_loss = self.train_loss_epoch / self.steps
        #     # avg_train_acc = self.train_acc_epoch / len(self.train_dataloader())
        #     # Append metrics to lists
        #     self.train_losses.append(avg_train_loss)
        #     # self.train_accuracies.append(avg_train_acc)
        #     # Reset metrics
        #     self.train_loss_epoch = 0.0
        #     self.train_acc_epoch = 0.0
        #     # Log metrics
        #     self.log('avg_train_loss', avg_train_loss, prog_bar=True,sync_dist=True)
        #     # self.log('avg_train_acc', avg_train_acc, prog_bar=True)

        def validation_step(self, batch, batch_idx):
            self.model.train()
            x, y = batch
            with torch.no_grad():
                loss_dict = self(x,y)
            loss = sum(v for v in loss_dict.values())
            loss = loss.mean()
            # Accumulate metrics for the epoch
            self.val_loss_epoch += loss.item()

            self.model.eval()
            with torch.no_grad():
                pred = self(x)
            for i in range(len(x)):
                bbox_pred, labels_pred, scores_pred = pred[i]['boxes'].cpu().numpy(), pred[i]['labels'].cpu().numpy(), pred[i]['scores'].cpu().numpy()
                bbox_gt, labels_gt = y[i]['boxes'].cpu().numpy(), y[i]['labels'].cpu().numpy()

                acc, mAP = compute_map(bbox_pred, labels_pred,scores_pred,bbox_gt,labels_gt,9)
                self.mAP_list.append(mAP)
                self.accuracy.append(acc)

            self.val_batch_mAP_list.append(np.array(self.mAP_list).mean())
            self.val_batch_accuracy.append(np.array(self.accuracy).mean())
            self.mAP_list = []
            self.accuracy = []

            return loss

        def on_validation_epoch_end(self):
            avg_train_loss = self.train_loss_epoch / self.steps
            avg_val_loss = self.val_loss_epoch / self.val_steps

            avg_train_map = np.array(self.batch_mAP_list).mean()
            avg_val_map = np.array(self.val_batch_mAP_list).mean()
            
            avg_train_acc = np.array(self.batch_accuracy).mean()
            avg_val_acc = np.array(self.val_batch_accuracy).mean()


            # Log metrics
            self.log('Train Loss', avg_train_loss, prog_bar=True,sync_dist=True)
            self.log('val_loss', avg_val_loss, prog_bar=True,sync_dist=True)
            self.log('Train Acc', avg_train_acc, prog_bar=True,sync_dist=True)
            self.log('Val Acc', avg_val_acc, prog_bar=True,sync_dist=True)
            self.log('Train mAP50', avg_train_map, prog_bar=True,sync_dist=True)
            self.log('val_map', avg_val_map, prog_bar=True,sync_dist=True)

            #update df
            self.df = self.df._append({"Epoch": self.current_epoch + 1, "Train_Loss": avg_train_loss, "Train_Accuracy": avg_train_acc, "Train_mAP": avg_train_map,
                        "Val_Loss":avg_val_loss, "Val_Accuracy":avg_val_acc, "Val_mAP":avg_val_map}, ignore_index=True)
            
            # Reset metrics
            self.val_loss_epoch = 0.0
            self.train_loss_epoch = 0.0

            # Append metrics to lists
            # self.val_losses.append(avg_val_loss)
            self.val_batch_mAP_list = [] 
            self.val_batch_accuracy = []
            self.batch_mAP_list = []
            self.batch_accuracy = []

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr,epochs=args.max_epochs,steps_per_epoch=self.steps)
            return [optimizer], [scheduler]

        def on_train_end(self):
            self.df.to_csv(f'{args.name}/metrics_1.csv',index = False)

    faster_model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    num_classes = 9
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    semseg = deeplabv3_resnet101(weight='DEFAULT',num_classes = 13)
    checkpoint = torch.load('/home/harishs/projects/def-akilan/harishs/objdet/kitti_segmentation_final/my_model-epoch=99-val_loss=0.061-Val_mIoU=0.558.ckpt')
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace('model.', '')
        new_state_dict[new_key] = v
    semseg.load_state_dict(new_state_dict)

    # Determine the parameters to freeze
    for param in list(semseg.parameters()):
        param.requires_grad = False

    model = combineModel(semseg,faster_model)

    train_data = CustomImageDataset(annot_dir='/home/harishs/projects/def-akilan/harishs/objdet/kitti/label_2',
                                img_dir='/home/harishs/projects/def-akilan/harishs/objdet/kitti/train_image',
                                transform=None,  # Add your image transformation if any
                                target_transform=None)
    
    val_data = CustomImageDataset(annot_dir='/home/harishs/projects/def-akilan/harishs/objdet/kitti/test_labels',
                            img_dir='/home/harishs/projects/def-akilan/harishs/objdet/kitti/test_image',
                            transform=None,  # Add your image transformation if any
                            target_transform=None)


    batch_size = args.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,collate_fn=collate,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,collate_fn=collate,drop_last=True)

    steps = len(train_loader)
    print(f'steps: {steps}')
    val_steps = len(val_loader)
    print(f'Val steps: {val_steps}')
    
    os.makedirs(args.name, exist_ok=True)
    net = DeepLabNet(model,steps,val_steps)

    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Metric to monitor
    dirpath=f'{args.name}/',  # Directory to save checkpoints
    filename='my_model-{epoch:02d}-{val_loss:.3f}-{val_map:.3f}',  # Checkpoint filename
    save_top_k=10,  # Save only the best model
    mode='min',  # Minimize the monitored metric
    )

    map_checkpoint_callback = ModelCheckpoint(
    monitor='val_map',  # Metric to monitor
    dirpath=f'{args.name}/',  # Directory to save checkpoints
    filename='my_model-{epoch:02d}-{val_loss:.3f}-{val_map:.3f}',  # Checkpoint filename
    save_top_k=10,  # Save only the best model
    mode='max',  # Minimize the monitored metric
    )
    
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0003,
    patience=8,
    verbose=True,
    mode='min')

    trainer = pl.Trainer(accelerator="gpu", devices= torch.cuda.device_count(), num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES")), strategy='ddp',
                          max_epochs=args.max_epochs, default_root_dir=f'/home/harishs/projects/def-akilan/harishs/objdet/{args.name}',enable_progress_bar=True,
                          callbacks=[TQDMProgressBar(refresh_rate=50),checkpoint_callback,map_checkpoint_callback]) 

    trainer.fit(net, train_loader,val_loader,ckpt_path='/home/harishs/projects/def-akilan/harishs/objdet/fasterdeeplabattn_234_v2/my_model-epoch=27-val_loss=0.078.ckpt')

if __name__ == '__main__':
    np.seed= 0
    main()





            

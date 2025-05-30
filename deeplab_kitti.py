import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
import torchvision
import torchvision.transforms as transforms
from cityscapes import CustomMaskDataset
from torch.utils.data import DataLoader
import argparse
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import pandas as pd
from deeplabseg import _fast_hist,scores,collate
import numpy as np
from torchvision import transforms as T
import os 
from torch.utils.data import random_split
from lightning.pytorch.callbacks import TQDMProgressBar,ModelCheckpoint,EarlyStopping
import warnings
import gc

parser = argparse.ArgumentParser(description='Deeplab Segmentation')
parser.add_argument('--lr', type = float, default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--name', type=str, default=4, help='')

warnings.filterwarnings('ignore')

def main():
    print("Starting...")

    model = deeplabv3_resnet101(weight=DeepLabV3_ResNet101_Weights,num_classes = 13)
    state_dict = torch.load('/home/harishs/projects/def-akilan/harishs/objdet/deeplab_final_imsize/my_model-epoch=10-val_loss=0.074-Val_mIoU=0.687.ckpt')
    new_state_dict = {}
    for k, v in state_dict['state_dict'].items():
        new_key = k.replace('model.', '')
        new_state_dict[new_key] = v
    print('Weights Loades', flush = True)

    args = parser.parse_args()

    class DeepLabNet(pl.LightningModule):
        def __init__(self,model,steps,val_steps):
            super(DeepLabNet, self).__init__()
            # Load the pre-trained DeepLab model
            self.model = model
            # Initialize lists to store metrics
            self.steps = steps
            self.val_steps = val_steps
            # self.train_losses = []
            # self.train_accuracies = []
            # self.val_losses = []
            # self.val_accuracies = []
            self.train_loss_epoch = 0.0
            # self.train_acc_epoch = 0.0
            self.val_loss_epoch = 0.0
            # self.val_acc_epoch = 0.0
            self.all_train_labels = []
            self.all_train_preds = []
            self.all_val_labels = []
            self.all_val_preds = []
            self.df = pd.DataFrame(columns=["Epoch", "Train_Loss", "Train_Pixel_Acc", "Train_Mean_Acc","Train_FW_IOU","Train_Mean_IoU",
                               "Val_Loss", "Val_Pixel_Acc", "Val_Mean_Acc","Val_FW_IOU","Val_Mean_IoU"])
        
        def forward(self, x):
            x = self.model(x)['out']
            return x
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            y_hat = y_hat.detach().cpu()
            y = y.detach().cpu()
            y_hat = torch.argmax(y_hat,dim=1)
            y = torch.argmax(y, dim = 1)
            # Accumulate metrics for the epoch
            self.train_loss_epoch += loss.item()

            self.all_train_labels.extend(y.cpu().numpy())
            self.all_train_preds.extend(y_hat.cpu().numpy())

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
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            y_hat = y_hat.detach().cpu()
            y = y.detach().cpu()
            y_hat = torch.argmax(y_hat,dim=1)
            y = torch.argmax(y, dim = 1)

            self.all_val_labels.extend(y.cpu().numpy())
            self.all_val_preds.extend(y_hat.cpu().numpy())
            # Accumulate metrics for the epoch
            self.val_loss_epoch += loss.item()
            return loss

        def on_validation_epoch_end(self):
            avg_train_loss = self.train_loss_epoch / self.steps
            # Compute average metrics for the epoch
            avg_val_loss = self.val_loss_epoch / self.val_steps
            train_score = scores(self.all_train_labels,self.all_train_preds,13)
            val_score = scores(self.all_val_labels,self.all_val_preds,13)

            # Append metrics to lists
            # self.val_losses.append(avg_val_loss)
            self.all_train_labels = [] 
            self.all_train_preds = []
            self.all_val_labels = []
            self.all_val_preds = []

            # Reset metrics
            self.val_loss_epoch = 0.0
            self.train_loss_epoch = 0.0

            # Log metrics
            self.log('Train Loss', avg_train_loss, prog_bar=True,sync_dist=True)
            self.log('val_loss', avg_val_loss, prog_bar=True,sync_dist=True)
            self.log('Train Acc', train_score['Mean Accuracy'], prog_bar=True,sync_dist=True)
            self.log('Val Acc', val_score['Mean Accuracy'], prog_bar=True,sync_dist=True)
            self.log('Train mIoU', train_score['Mean IoU'], prog_bar=True,sync_dist=True)
            self.log('Val_mIoU', val_score['Mean IoU'], prog_bar=True,sync_dist=True)

            #update df
            self.df = self.df._append({"Epoch": self.current_epoch + 1, "Train_Loss": avg_train_loss, "Train_Pixel_Acc": train_score['Pixel Accuracy'], "Train_Mean_Acc": train_score['Mean Accuracy'],
                    "Train_FW_IOU": train_score['Frequency Weighted IoU'],"Train_Mean_IoU":train_score["Mean IoU"],"Val_Loss":avg_val_loss, 
                    "Val_Pixel_Acc":val_score["Pixel Accuracy"], "Val_Mean_Acc": val_score["Mean Accuracy"],"Val_FW_IOU" : val_score["Frequency Weighted IoU"],
                    "Val_Mean_IoU":val_score['Mean IoU']}, ignore_index=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr,epochs=args.max_epochs,steps_per_epoch=self.steps)
            return [optimizer], [scheduler]

        def on_train_end(self):
            self.df.to_csv(f'{args.name}/metrics.csv',index = False)
            # # Create a DataFrame from the metrics lists
            # metrics_df = pd.DataFrame({
            #     'train_loss': self.train_losses,
            #     'train_acc': self.train_accuracies,
            #     'val_loss': self.val_losses,
            #     'val_acc': self.val_accuracies
            # })
            # # Save the DataFrame to a CSV file
            # metrics_df.to_csv('metrics.csv', index=False)


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
        T.Resize((402, 1333))  # Resize the input image to the expected size
            # Convert the image to a PyTorch tensor
    ])

    data = CustomMaskDataset(mask_dir='/home/harishs/projects/def-akilan/harishs/objdet/training/semantic',
                            img_dir='/home/harishs/projects/def-akilan/harishs/objdet/training/image_2',
                            ignoreeval_list=ignoreeval_list,
                            eqv_dict=eqv_dict,
                            transform=transform)
    
    train_split = int(len(data)*0.90)
    val_split = int(len(data)-train_split)
    print(train_split,val_split)

    train_dataset, val_dataset = random_split(data,[train_split,val_split])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=collate,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,collate_fn=collate)
    gc.collect()
    torch.cuda.empty_cache()

    steps = len(train_loader)
    val_steps = len(val_loader)

    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Metric to monitor
    dirpath=f'{args.name}/',  # Directory to save checkpoints
    filename='my_model-{epoch:02d}-{val_loss:.3f}-{Val_mIoU:.3f}',  # Checkpoint filename
    save_top_k=6,  # Save only the best model
    mode='min',  # Minimize the monitored metric
    )

    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0003,
    patience=8,
    verbose=True,
    mode='min')


    # class CityScapesDataModule(pl.LightningDataModule):
    #     def __init__(self,ignoreeval_list,eqv_dict, image_dir: str = "./", mask_dir: str = "./"):
    #         super().__init__()
    #         self.image_dir = image_dir
    #         self.mask_dir = mask_dir
    #         self.ignoreeval_list=ignoreeval_list,
    #         self.eqv_dict=eqv_dict,

    #         self.transform=T.Compose([
    #         T.ToTensor(), 
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         T.Resize((256,512))  # Resize the input image to the expected size
    #             # Convert the image to a PyTorch tensor
    #     ])


    #     def setup(self, stage: str):
    #         # Assign train/val datasets for use in dataloaders
    #         if stage == "fit":
    #             cityscapes_full = CityScapesDataset(self.image_dir,self.mask_dir,ignoreeval_list=self.ignoreeval_list,
    #                             eqv_dict=self.eqv_dict,
    #                             transform=self.transform)
                
    #             train_split = int(len(cityscapes_full)*0.85)
    #             val_split = int(len(cityscapes_full)-train_split)
    #             self.cityscapes_train, self.cityscapes_val = random_split(cityscapes_full, [train_split, val_split])


    #     def train_dataloader(self):
    #         return DataLoader(self.cityscapes_train, batch_size=args.batch_size)

    #     def val_dataloader(self):
    #         return DataLoader(self.cityscapes_val, batch_size=args.batch_size)


    # dm = CityScapesDataModule(ignoreeval_list,eqv_dict,'/home/harishs/projects/def-akilan/harishs/objdet/leftImg8bit/train',
    #                           '/home/harishs/projects/def-akilan/harishs/objdet/gtFine/train')

    os.makedirs(args.name, exist_ok=True)
    net = DeepLabNet(model,steps,val_steps)

    trainer = pl.Trainer(accelerator="gpu", devices= torch.cuda.device_count(), num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES")), 
                        strategy='ddp',max_epochs=args.max_epochs, default_root_dir='/home/harishs/projects/def-akilan/harishs/objdet/{args.name}',
                        enable_progress_bar=True,callbacks=[TQDMProgressBar(refresh_rate=50),checkpoint_callback,early_stop_callback]) 

    trainer.fit(net, train_loader,val_loader)

if __name__ == '__main__':
    main()

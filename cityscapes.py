import os
import torch 
import torch.nn as nn
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
import re
import matplotlib.patches as patches


class CityScapesDataset(Dataset):
    def __init__(self, mask_dir, img_dir, ignoreeval_list,eqv_dict,transform=None):
        self.cities = os.listdir(img_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.ignoreeval_list = ignoreeval_list
        self.eqv_dict = eqv_dict
        self.image_files = []
        for city in self.cities:
            city_path = os.path.join(self.img_dir, city)
            for img_path in os.listdir(city_path):
                self.image_files.append(os.path.join(city,img_path))

    def __len__(self):
        return len(self.image_files)
    
    def process_mask(self, mask):
        # Convert PIL image to numpy array
        mask_np = np.array(mask)

        # Replace ignoreeval_list with 255
        mask_np[np.isin(mask_np, self.ignoreeval_list)] = 255

        # for mask_id, train_id in self.eqv_dict.items():
        #     mask_np[(mask_np == mask_id)] = train_id

        # Map mask values according to eqv_dict
        map_func = np.vectorize(lambda x: self.eqv_dict.get(x, x))
        mask_mapped = map_func(mask_np)

        # Convert numpy array back to PIL image
        mask_mapped = Image.fromarray(mask_mapped.astype(np.uint8))
        return mask_mapped
    
    def one_hot_encoder(self, bitmask):
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
        # function to create a one hot encoded bitmask for catergorical cross entropy loss training

        height, width = bitmask.shape

        one_hot_vector_list = []

        for key in color_dict:
            object_mask = np.zeros((height, width))
            object_loc = bitmask == key
            object_mask[object_loc] = 1 # set the location where where key is to 1
            one_hot_vector_list.append(object_mask)

        return np.dstack(one_hot_vector_list) 

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path)
        mask_path = os.path.join(self.mask_dir, re.sub('_leftImg8bit.png','_gtFine_labelIds.png',self.image_files[idx]))
        mask = Image.open(mask_path)

        mask = self.process_mask(mask)

        if self.transform:
            image = self.transform(image)

        mask = T.Resize((375,575),interpolation=Image.NEAREST)(mask)
        mask = np.array(mask)
        mask = self.one_hot_encoder(mask)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2,0,1)

        return image, mask
    
class CustomMaskDataset(Dataset):
    def __init__(self, mask_dir, img_dir, ignoreeval_list,eqv_dict,transform=None):
        self.image_files = os.listdir(img_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.ignoreeval_list = ignoreeval_list
        self.eqv_dict = eqv_dict

    def __len__(self):
        return len(self.image_files)
    
    def process_mask(self, mask):
        # Convert PIL image to numpy array
        mask_np = np.array(mask)

        # Replace ignoreeval_list with 255
        mask_np[np.isin(mask_np, self.ignoreeval_list)] = 255

        # for mask_id, train_id in self.eqv_dict.items():
        #     mask_np[(mask_np == mask_id)] = train_id

        # Map mask values according to eqv_dict
        map_func = np.vectorize(lambda x: self.eqv_dict.get(x, x))
        mask_mapped = map_func(mask_np)

        # Convert numpy array back to PIL image
        mask_mapped = Image.fromarray(mask_mapped.astype(np.uint8))
        return mask_mapped
    
    def one_hot_encoder(self, bitmask):
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
        # function to create a one hot encoded bitmask for catergorical cross entropy loss training

        height, width = bitmask.shape

        one_hot_vector_list = []

        for key in color_dict:
            object_mask = np.zeros((height, width))
            object_loc = bitmask == key
            object_mask[object_loc] = 1 # set the location where where key is to 1
            one_hot_vector_list.append(object_mask)

        return np.dstack(one_hot_vector_list) 

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        mask = self.process_mask(mask)

        if self.transform:
            image = self.transform(image)

        mask = T.Resize((402, 1333),interpolation=Image.NEAREST)(mask)
        mask = np.array(mask)
        mask = self.one_hot_encoder(mask)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2,0,1)

        return image, mask
    

# color_dict = {0: (0.502, 0.251, 0.502),
# 1: (0.957, 0.137, 0.910),
# 2: (0.275, 0.275, 0.275),
# 3: (0.275, 0.510, 0.706),
# 4: (0.863, 0.078, 0.235),
# 5: (1.0, 0.0, 0.0),
# 6: (0.0, 0.0, 0.557),
# 7: (0.0, 0.0, 0.275),
# 8: (0.0, 0.235, 0.392),
# 9: (0.0, 0.314, 0.392),
# 10: (0.0, 0.0, 0.920),
# 11: (0.467, 0.043, 0.125),
# 12: (0, 0, 0)}


# eqv_dict = {7: 0,       #sets mask_ids to their approriate train_ids that will be evaluated after prediction 
#             8: 1,
#             11: 2,
#             12: 2,
#             13: 2,
#             23: 3,
#             24: 4,
#             25: 5,
#             26: 6,
#             27: 7,
#             28: 8,
#             31: 9,
#             32: 10,
#             33: 11,
#             255: 12}


# # Initialize a list containing all the object classes to identify the class from the index
# object_class = ["road",
#         "sidewalk",
#         "background",
#         "sky",
#         "person",
#         "rider",
#         "car",
#         "truck",
#         "bus",
#         "train",
#         "motorcycle",
#         "bicycle",
#         "unknown"]

# ignoreeval_list = np.array([0, 4, 5, 6, 9, 10, 14, 
#                             15, 16, 17, 18, 19, 20, 21, 22, 29, 30,34]) #mask_ids in the cityscapes bitmasks that are not evaluated
# transform = T.Compose([
#     T.ToTensor(), 
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     T.Resize((512,1024))  # Resize the input image to the expected size
#         # Convert the image to a PyTorch tensor
# ])
    
# data = CityScapesDataset('/home/harishs/projects/def-akilan/harishs/objdet/gtFine/train',
#                          '/home/harishs/projects/def-akilan/harishs/objdet/leftImg8bit/train',
#                         ignoreeval_list=ignoreeval_list,
#                         eqv_dict=eqv_dict,
#                         transform=transform)

# def collate(batch):
#     # Unpack the batch into separate lists for images and masks
#     images, masks = zip(*batch)
    
#     # Convert the list of images into a batch tensor
#     images = torch.stack(images, dim=0)
    
#     # Convert the list of masks into a batch tensor
#     masks = torch.stack(masks, dim=0)
    
#     return images, masks

# train_loader = DataLoader(data, batch_size=8,collate_fn=collate,shuffle=True)

# def unnormalize(image_tensor, mean, std):
#     """
#     Unnormalize a normalized image tensor.

#     Args:
#         image_tensor (torch.Tensor): Normalized image tensor.
#         mean (sequence): Sequence of means for each channel.
#         std (sequence): Sequence of standard deviations for each channel.

#     Returns:
#         torch.Tensor: Unnormalized image tensor.
#     """
#     # Create a copy of the image tensor
#     unnormalized_image = image_tensor.clone()

#     # Iterate over each channel and unnormalize
#     for channel, mean_value, std_value in zip(unnormalized_image, mean, std):
#         channel.mul_(std_value).add_(mean_value)

#     return unnormalized_image

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# def colorize_bitmask(bitmask, color_dict):
#     height, width = bitmask.shape
#     colorized_image = np.zeros((height, width, 3)) # create a 0 array that is of the shape of the bitmask

#     for key, color in color_dict.items():
#         mask = (bitmask == key)
#         colorized_image[mask]= color # assign the color based on the key
#     return colorized_image # return colorized image

# for imgs, masks in train_loader:
#     print(imgs.shape)
#     print(masks.shape)
#     masks = torch.argmax(masks, dim = 1)
#     imgs = imgs[:5,:,:,:]
#     masks = masks[:5,:,:]

#     fig, axes = plt.subplots(5,2, figsize = (20,10))
#     for i,(image,gt_mask) in enumerate(zip(imgs,masks)):
#         image = unnormalize(image, mean, std)
#         ax = axes[i]
#         image = image.cpu().permute(1,2,0).numpy()
#         gt_mask = gt_mask.cpu().numpy()
#         gt_mask = colorize_bitmask(gt_mask,color_dict)
#         ax[0].imshow(image)
#         ax[1].imshow(gt_mask)

#         ax[0].axis('off')
#         ax[1].axis('off')
    
#     axes[0][0].set_title('Images',fontsize = 25)
#     axes[0][1].set_title('Ground Truth',fontsize = 25)

#     break
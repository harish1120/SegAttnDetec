import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as T
from FCOS.fcos_structure.boxlist import BoxList
from FCOS.fcos_structure.image_list import ImageList, to_image_list


def label_data_record(record):
    """
    Given a string 'record', this will
    output a dictionary of the appropriate labels
    and data types.
    """
    record_list = record.split(" ")
    record_dict = {
        "type":record_list.pop(0),
        "truncated":float(record_list.pop(0)),
        "occluded":int(record_list.pop(0)),
        "alpha":float(record_list.pop(0)),
        "left":float(record_list.pop(0)),
        "bottom":float(record_list.pop(0)),
        "right":float(record_list.pop(0)),
        "top":float(record_list.pop(0)),
        "height":float(record_list.pop(0)),
        "width":float(record_list.pop(0)),
        "length":float(record_list.pop(0)),
        "x":float(record_list.pop(0)),
        "y":float(record_list.pop(0)),
        "z":float(record_list.pop(0)),
    }
    return record_dict 


def extract_bbox(labeled_record):
    """
    Given a labeled record from the training data, 
    this will output a 3-tuple with the bounding box parameters.
    This output can be passed into Rectangle()
    """
    type = {'Car':1,
            'Pedestrian':2,
            'Truck':3,
            'Cyclist':4,
            'Person_sitting':5,
            'Van':6,
            'Tram':7,
            'Misc':8}
    
    if labeled_record['type'] != 'DontCare':
        width = labeled_record["right"] - labeled_record["left"]
        height = labeled_record["bottom"] - labeled_record["top"]
        bbox = [labeled_record["left"],labeled_record["bottom"],labeled_record["right"],labeled_record["top"],int(type[labeled_record["type"]])]
        return bbox

def extract_bbox_fcos(labeled_record):
    """
    Given a labeled record from the training data, 
    this will output a 3-tuple with the bounding box parameters.
    This output can be passed into Rectangle()
    """
    type = {'Car':0,
            'Pedestrian':1,
            'Truck':2,
            'Cyclist':3,
            'Person_sitting':4,
            'Van':5,
            'Tram':6,
            'Misc':7,
            'DontCare':8}
    
    width = labeled_record["right"] - labeled_record["left"]
    height = labeled_record["bottom"] - labeled_record["top"]
    bbox = [labeled_record["left"],labeled_record["bottom"],labeled_record["right"],labeled_record["top"],int(type[labeled_record["type"]])]
    return bbox    

def generate_targets_kitti(bboxes, labels, image_size, strides, num_classes):
    h, w = image_size
    cls_targets, reg_targets, center_targets = [], [], []

    for stride in strides:
        feature_h, feature_w = h // stride, w // stride
        cls_target = torch.zeros((feature_h, feature_w, num_classes), dtype=torch.float32)
        reg_target = torch.zeros((feature_h, feature_w, 4), dtype=torch.float32)
        center_target = torch.zeros((feature_h, feature_w), dtype=torch.float32)
        
        for bbox, label in zip(bboxes, labels):
            # Compute the center of the bounding box
            x_min, y_min, x_max, y_max = bbox
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            
            # Map the center to the feature map
            fx, fy = int(cx / stride), int(cy / stride)
            if 0 <= fx < feature_w and 0 <= fy < feature_h:
                cls_target[fy, fx, label] = 1.0
                reg_target[fy, fx, 0] = cx - x_min
                reg_target[fy, fx, 1] = cy - y_min
                reg_target[fy, fx, 2] = x_max - cx
                reg_target[fy, fx, 3] = y_max - cy
                
                left, top, right, bottom = reg_target[fy, fx]
                centerness = torch.sqrt(
                    min(left, right) * min(top, bottom) /
                    (max(left, right) * max(top, bottom) + 1e-8)
                )
                center_target[fy, fx] = centerness

        cls_targets.append(cls_target.permute(2, 0, 1))  # Channel-first format
        reg_targets.append(reg_target)
        center_targets.append(center_target)

    return cls_targets, reg_targets, center_targets

class CustomImageDataset(Dataset):
        def __init__(self, annot_dir, img_dir, transform=None, target_transform=None):
            self.image_files = os.listdir(img_dir)
            self.img_dir = img_dir
            self.annotation = annot_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.image_files)
        
        def resize_box(self,box,image_size_old, image_size_new):
            o_x = image_size_old[0]
            o_y = image_size_old[1]

            n_x = image_size_new[0]
            n_y = image_size_new[1]

            scale_x = n_x/o_x
            scale_y = n_y/o_y
            box[0] = box[0]*scale_x
            box[1] = box[1]*scale_y
            box[2] = box[2]*scale_x
            box[3] = box[3]*scale_y
            return box

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.image_files[idx])

            annot_file = self.image_files[idx].replace('.png','.txt')
            annot_path = os.path.join(self.annotation,annot_file)
            image = Image.open(img_path)
            o_size = image.size
            trans_resize = T.Resize((128,128))
            image = trans_resize(image)
            n_size = image.size
            with open(annot_path) as f:
                records = f.readlines()
            bbox = []
            labels = []
            for record in records:  
                single_record = label_data_record(record)
                bbox_params = extract_bbox(single_record)
                if bbox_params is not None:
                    label = int(bbox_params[-1])
                    bbox_params=bbox_params[:-1]
                    bbox_params = self.resize_box(bbox_params,o_size,n_size)
                    bbox.append(bbox_params)
                    labels.append(label)
            target = {}
            labels = torch.tensor(labels)
            target['boxes'] = torch.Tensor(bbox)
            target['labels'] = labels
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return T.ToTensor()(image), target
        

class CustomImageDataset_FCOS(Dataset):
        def __init__(self, annot_dir, img_dir, transform=None, target_transform=None):
            self.image_files = os.listdir(img_dir)
            self.img_dir = img_dir
            self.annotation = annot_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.image_files)
        
        def resize_box(self,box,image_size_old, image_size_new):
            o_x = image_size_old[0]
            o_y = image_size_old[1]

            n_x = image_size_new[0]
            n_y = image_size_new[1]

            scale_x = n_x/o_x
            scale_y = n_y/o_y
            box[0] = box[0]*scale_x
            box[1] = box[1]*scale_y
            box[2] = box[2]*scale_x
            box[3] = box[3]*scale_y
            return box

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.image_files[idx])

            annot_file = self.image_files[idx].replace('.png','.txt')
            annot_path = os.path.join(self.annotation,annot_file)
            image = Image.open(img_path)
            x,y = image.size
            with open(annot_path) as f:
                records = f.readlines()
            bbox = []
            labels = []
            for record in records:  
                single_record = label_data_record(record)
                bbox_params = extract_bbox_fcos(single_record)
                if bbox_params is not None:
                    label = int(bbox_params[-1])
                    bbox_params=bbox_params[:-1]
                    bbox.append(bbox_params)
                    labels.append(label)
            # target = {}
            t = BoxList(bbox, (x, y))
            labels = torch.Tensor(labels)
            t.add_field('labels',labels)
            # print(t)
            # labels = torch.tensor(labels)
            # target['boxes'] = torch.Tensor(bbox)
            # target['labels'] = labels
            # if self.transform:
            #     image = self.transform(image)
            # if self.target_transform:
            #     label = self.target_transform(label)
            return T.ToTensor()(image), t

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from torch.utils.data import random_split
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def collate(data):
    #     # Separate images and targets
    #     imgs = [item[0] for item in data]
    #     targets = [{'boxes': item[1]['boxes'],
    #                 'labels': item[1]['labels']} for item in data]

    #     # Move images and targets to the device
    #     imgs = torch.stack(imgs).to(device)
    #     for target in targets:
    #         target['boxes'] = target['boxes'].to(device)
    #         target['labels'] = target['labels'].to(device)

    #     return imgs, targets
    def collate_fn(batch, device):
        """
        Custom collate function for FCOS dataset.
        
        Args:
            batch (list): A list of tuples, where each tuple contains:
                        (image, BoxList(target)).
            device (torch.device): The device to move the data to.
        
        Returns:
            Tuple[Tensor, list[BoxList]]: Batched images and targets.
        """
        # Separate images and BoxList targets
        images, targets = zip(*batch)

        # Convert images to a single batched tensor and move to device
        images = to_image_list(images)
        images = images.to(device)


        # Move BoxList fields (bbox, labels) to the device
        targets = [target.to(device) for target in targets]

        return images, targets

    dataset = CustomImageDataset_FCOS(annot_dir='/home/harishs/scratch/kitti/label_2',
                            img_dir='/home/harishs/scratch/kitti/train_image',
                            transform=None,  # Add your image transformation if any
                            target_transform=None)

    batch_size = 4
    train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, collate_fn=lambda batch: collate_fn(batch, device), drop_last=True)

    unique_labels = set() 

    for images,targets in train_loader:
        print(images.image_sizes)
        print([target.bbox.shape for target in targets])
        print([target.get_field('labels') for target in targets])
        break

    for i,(images,targets) in enumerate(train_loader):
        for target in targets:
            unique_labels.update(target.get_field('labels').tolist())
        if i == 100:
            break

    unique_labels = sorted(unique_labels)
    print(f"Unique labels: {unique_labels}")


    for images, targets in train_loader:
        for target in targets:
            target.to('cpu')
        print(images.image_sizes)
        print([target.bbox.shape for target in targets])
        print([target.get_field('labels') for target in targets])

        # Convert tensor images to NumPy arrays and transpose to (H, W, C) format
        images_numpy = [image.permute(1, 2, 0).cpu().numpy()for image in images.tensors]

        # # Get ground truth boxes and labels from the targets
        gt_boxes = targets[2].bbox
        gt_labels = targets[2].get_field('labels')
        fig, ax = plt.subplots(1)
        ax.imshow(images_numpy[2])

        # # Visualize predicted boxes and labels
        for box, label in zip(gt_boxes, gt_labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], str(label), color='r')

        plt.savefig(f'testimage_{1}.png')
        plt.close()
        break
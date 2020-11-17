import torch
import os
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader, DatasetFolder
from PIL import Image
from utils import mosaic, demosaic, rand_rgb_image
import cv2
import numpy as np

full_path = "/home/michelexie/GithubLocal/digital_chirality/full_dataset/"
# def get_transform(image_size, crop, crop_size):
def get_transform():
    # Create train/valid transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_transforms = {
        'train': train_transform,
        'test': valid_transform,
    }

    # assert crop in ["none", 'random_crop_inside_boundary']
    #
    # transforms_list_train = []
    # transforms_list_test = []
    #
    # if crop == "none":
    #     print("No cropping to the images.")
    # elif crop == "random_crop_inside_boundary":
    #     boundary = image_size - 32 # 16 pixels boundary
    #     print(f"First performing a center crop of size {boundary} to avoid boundary")
    #     transforms_list_train += [
    #         transforms.CenterCrop(boundary),
    #         transforms.RandomCrop(crop_size)
    #     ]
    #     transforms_list_test += [
    #         transforms.CenterCrop(boundary),
    #         transforms.RandomCrop(crop_size)
    #     ]
    #
    # transforms_list_train += [transforms.ToTensor()]
    # transforms_list_test += [transforms.ToTensor()]

    # data_transforms = {
    #     'train': transforms.Compose(transforms_list_train),
    #     'test': transforms.Compose(transforms_list_test),
    # }
    return data_transforms

def get_dataloaders(
                    batch_size=16,
                    num_workers=4):
    '''
        Return a factory of PyTorch dataset/dataloader
    '''
    # data_transform = get_transform(image_size, crop, crop_size) # A dict with 'train' 'test'
    data_transform = get_transform()  # A dict with 'train' 'test'

    train_dataset = ChiralDataset(full_path,
                                  data_transform['train'])
    val_dataset = ChiralDataset(full_path,
                                  data_transform['test'])

    lengths = [round(len(train_dataset) * 0.8), round(len(train_dataset) * 0.2)]
    # print(lengths)
    # print(len(train_dataset))

    train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths)
    _, val_dataset = torch.utils.data.random_split(val_dataset, lengths)

    return {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=False),
        'val' : torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=False),
    }

# class ChiralDataset(torch.utils.data.Dataset):
#     def __init__(self,
#                  size,
#                  transform,
#                  image_size,
#                  image_type,
#                  image_pattern,
#                  demosaic_algo,
#                  bayer_pattern,
#                  jpeg_coeff=25):
#         """A dataset that contains randomly generated images with random flip
#             Args:
#             size - the size of dataset
#             transform - the transformation to be applied to images (e.g. random crop)
#             image_size - The size of image
#             image_type - the type of image (original/demosaic/jpeg/both)
#             image_pattern - the distribution of images
#             demosaic_algo - If the image undergone demosaicing step, then use this demosaic algorithm
#             bayer_pattern - If the image undergone demosaicing step, then use this bayer grid pattern
#             jpeg_coeff - The jpeg compression coefficient if using JPEG-based image_type.
#         """
#         self.class_names = ["flipped", "original"]
#         self.size = size
#
#         self.horizontalFlip = torchvision.transforms.RandomHorizontalFlip(p=1)
#         self.transform = transform
#         self.image_type = image_type
#         self.image_size = image_size
#         self.image_pattern = image_pattern
#         self.demosaic_algo = demosaic_algo
#         self.bayer_pattern = bayer_pattern
#
#         self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_coeff]
#
#         self.shared_batch_base_seed = 0 # Should be the epoch number
#
#     def __getitem__(self, index):
#         # original STRAT
#         # np.random.seed(self.shared_batch_base_seed * self.size + int(index/2))
#         # image_original = rand_rgb_image(self.image_size, self.image_pattern)# numpy.ndarray
#         # # print(type(image_original)) # numpy.ndarray
#         # if self.image_type == 'original':
#         #     img = image_original
#         # elif self.image_type == 'jpeg':
#         #     img = cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR)
#         #     _, img = cv2.imencode('.jpg', img, self.encode_param)
#         #     img = cv2.imdecode(img, 1)
#         #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # else:
#         #     image_demosaiced = demosaic(
#         #         mosaic(image_original, pattern=self.bayer_pattern),
#         #         pattern=self.bayer_pattern,
#         #         algo=self.demosaic_algo
#         #     ).astype('uint8')
#         #     if self.image_type == 'demosaic':
#         #         img = image_demosaiced
#         #     elif self.image_type == 'both':
#         #         both_new = cv2.cvtColor(image_demosaiced, cv2.COLOR_RGB2BGR)
#         #         _, both_new = cv2.imencode('.jpg', both_new, self.encode_param)
#         #         both_new = cv2.imdecode(both_new, 1)
#         #         img = cv2.cvtColor(both_new, cv2.COLOR_BGR2RGB)
#         # sample = Image.fromarray(img)
#         #
#         # if index % 2 == 0:
#         #     sample = self.horizontalFlip(sample)
#         #     label = 0 # Flip is 0
#         # else:
#         #     label = 1 # Original is 1
#         # original END
#
#
#         # NEW START
#         # np.random.seed(self.shared_batch_base_seed * self.size + int(index/2))
#         # image_original = rand_rgb_image(self.image_size, self.image_pattern)# numpy.ndarray
#         left_path = "/home/michelexie/GithubLocal/digital_chirality/face/split_left/"  # change this dirpath.
#         listdir_left = os.listdir(left_path)
#         right_path = "/home/michelexie/GithubLocal/digital_chirality/face/split_right/"  # change this dirpath.
#         listdir_right = os.listdir(right_path)
#         for i in listdir_left:
#             if i.split('.')[1] == "jpg":  # the format of img.
#                 filepath_left = os.path.join(left_path, i)
#                 img_left_original = cv2.imread(filepath_left)
#                 img = cv2.cvtColor(img_left_original, cv2.COLOR_RGB2BGR)
#                 _, img = cv2.imencode('.jpg', img, self.encode_param)
#                 img = cv2.imdecode(img, 1)
#                 img_left = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         for i in listdir_right:
#             if i.split('.')[1] == "jpg":  # the format of img.
#                 filepath_right = os.path.join(right_path, i)
#                 img_right_original = cv2.imread(filepath_right)
#                 img = cv2.cvtColor(img_right_original, cv2.COLOR_RGB2BGR)
#                 _, img = cv2.imencode('.jpg', img, self.encode_param)
#                 img = cv2.imdecode(img, 1)
#                 img_right = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#
#         if index % 2 == 0:
#             # sample = self.horizontalFlip(sample)
#             left_sample = Image.fromarray(img_left)
#             sample = left_sample
#             label = 0 # LEFT is 0
#
#         else:
#             right_sample = Image.fromarray(img_right)
#             sample = right_sample
#             label = 1 # RIGHT is 1
#         # NEW END
#
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample, label
#
#     def __len__(self):
#         return self.size # need to confirm



class ChiralDataset(torch.utils.data.Dataset):
    def __init__(self,
                 full_path,
                 transform, #keep
                 ):
        """A dataset that contains randomly generated images with random flip
            Args:
            transform - the transformation to be applied to images (e.g. random crop)
        """
        self.class_names = ["left", "right"]
        self.transform = transform
        self.full_path = full_path
        # self.right_path = right_path
        self.listdir_full = os.listdir(self.full_path)
        # self.listdir_right = os.listdir(self.right_path)


    def cv2_img(self, filepath):
        img_original = cv2.imread(filepath)
        img = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        # _, img = cv2.imencode('.jpg', img, self.encode_param)
        # img = cv2.imdecode(img, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        self.filepath = os.path.join(self.full_path, self.listdir_full[index - 1])
        self.img = self.cv2_img(self.filepath)
        # self.filepath_right = os.path.join(self.right_path, self.listdir_right[index - 1])
        # self.img_right = self.cv2_img(self.filepath_right)
        self.img = Image.fromarray(self.img)
        if index % 2 == 0:
            label = 0 # left is 0
        else:
            label = 1 # right is 1
        # self.img_right = Image.fromarray(self.img_right)
        # self.left_sample_ = (self.img_left, 0)
        # self.right_sample_ = (self.img_right, 1)

        # self.sample_ = self.left_sample_ + self.right_sample_
        # self.sample = self.sample_[::2]
        # self.label = self.sample_[1::2]

        if self.transform:
            sample = self.transform(self.img)
        # assert 0
        return sample, label
    def __len__(self):
        return 15339
        # return len(self.sample)
        # return self.size #len(dataset_size)



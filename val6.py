# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import torch.utils.data
from albumentations.augmentations import functional as F
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm
from PIL import Image

import archs
from dataset import Dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []

        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])

        mask = np.dstack(mask)

        # print(img.shape)
        # print(mask.shape)
        # print(gradient.shape)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)  # 这个包比较方便，能把mask也一并做掉

            img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
            # augmented = self.transform(image=img,mask=gradient)#这个包比较方便，能把mask也一并做掉
            original_height, original_width = img.shape[:2]

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        # print(gradient.size)

        return img, mask, {'img_id': img_id, 'original_width': int(original_width),
                           'original_height': int(original_height)}


class CustomDataset(Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
            original_height, original_width = img.shape[:2]

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        return img, {'img_id': img_id, 'original_width': int(original_width), 'original_height': int(original_height)}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="dsb2018_96_UNet_woDS",
                        help='model name')
    parser.add_argument('--input_path', default="inputs/test/images",
                        help='path to input images')
    parser.add_argument('--output_path', default="inputs/out/images",
                        help='path to save output images')

    args = parser.parse_args()

    return args
# 在 main 函数中使用 CustomDataset
def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'FPN':
        blocks = [2, 4, 23, 3]
        model = archs.__dict__[config['arch']](blocks, config['num_classes'], back_bone='resnet101')
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join(args.input_path, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    model.load_state_dict(torch.load('models/%s/model.pth' % args.name))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ], )

    val_dataset = CustomDataset(
        img_ids=img_ids,
        img_dir=args.input_path,
        img_ext=config['img_ext'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    for c in range(config['num_classes']):
        os.makedirs(os.path.join(args.output_path, str(c)), exist_ok=True)

    with torch.no_grad():
        for input, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    binary_output = (output[i, c] > 0.5).astype('uint8')  # Binarize

                    # Load original image to get its size
                    original_image = cv2.imread(
                        os.path.join(args.input_path, meta['img_id'][i] + config['img_ext']))
                    original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)

                    # Resize binary output to original image size
                    resized_output = cv2.resize(binary_output, original_size, interpolation=cv2.INTER_NEAREST)

                    # Convert to PIL Image and save as 1-bit PNG
                    img = Image.fromarray(resized_output * 255).convert('1')
                    img.save(os.path.join(args.output_path, str(c), meta['img_id'][i] + '.png'))

    # plot_examples(val_dataset, model, num_examples=3)
    torch.cuda.empty_cache()




if __name__ == '__main__':
    main()


# python val6.py --input_path <输入图片的路径> --output_path <输出图片的路径>
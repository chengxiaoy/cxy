from glob import glob
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from random import choice, sample
import numpy as np
import PIL
from PIL import Image
import torchvision
import torch

mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
mean_rgb = np.array([131.0912, 103.8827, 91.4953])


def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    # img -= mean_rgb
    img = img.transpose(2, 0, 1)  # C x H x W

    # img = torchvision.transforms.ToTensor()(img)
    # img = torch.from_numpy(img).float()
    img = torch.from_numpy(img).float()
    return img


def loader(image_file, split, argument=False):
    img = Image.open(image_file)
    img = torchvision.transforms.Resize(197)(img)
    if argument:
        # img = torchvision.transforms.Resize(256)(img)
        if split == 'train':
            trans = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.RandomRotation(90),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomVerticalFlip(0.5),
            ])
            img = trans(img)

        else:
            img = torchvision.transforms.CenterCrop(224)(img)
    img = np.array(img, dtype=np.uint8)
    return transform(img)


def get_data():
    train_file_path = "Faces_in_the_Wild/train_relationships.csv"
    train_folders_path = "Faces_in_the_Wild/train/"
    val_famillies = "F09"

    all_images = glob(train_folders_path + "*/*/*.jpg")
    all_images = [x.replace('\\', '/') for x in all_images]
    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]
    return train, train_person_to_images_map, val, val_person_to_images_map


class FaceDataSet(Dataset):
    def __init__(self, relations, label_images_map, kind, argument=False):
        self.relations = relations
        self.label_images_map = label_images_map
        self.kind = kind
        self.length = self.get_length()
        self.argument = argument
        self.same = True

    def __getitem__(self, index):

        # should_same = self.same
        # self.same = not self.same
        should_same = index % 2
        p1, p2 = choice(self.relations)
        if should_same:
            img1 = loader(choice(self.label_images_map[p1]), self.kind, self.argument)
            img2 = loader(choice(self.label_images_map[p2]), self.kind, self.argument)
            return img1, img2, torch.Tensor([1])
        else:
            while True:
                p3, p4 = choice(self.relations)
                if p1 != p4 and (p1, p4) not in self.relations and (p4, p1) not in self.relations:
                    img1 = loader(choice(self.label_images_map[p1]), self.kind, self.argument)
                    img2 = loader(choice(self.label_images_map[p4]), self.kind, self.argument)
                    return img1, img2, torch.Tensor([0])

    def get_length(self):
        length = 0
        for key in self.label_images_map:
            length = length + len(self.label_images_map[key])
        return length

    def __len__(self):
        # if self.kind == 'train':
        #     return self.length
        return self.length

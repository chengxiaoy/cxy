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
from tricks import tricks
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random

mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
mean_rgb = np.array([131.0912, 103.8827, 91.4953])


def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    # img -= mean_rgb
    img = img.transpose(2, 0, 1)  # C x H x W

    # img = torchvision.transforms.ToTensor()(img)
    # img = img.half()
    #
    img = torch.from_numpy(img).float()
    return img


def loader(image_file, split, argument=False):
    img = Image.open(image_file)
    if argument:
        # img = torchvision.transforms.Resize(256)(img)
        if split == 'train':
            # img = torchvision.transforms.Resize(197)(img),
            trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize(197),
                # torchvision.transforms.RandomGrayscale(p=0.2),
                # torchvision.transforms.RandomRotation(90),
                # torchvision.transforms.RandomHorizontalFlip(0.5),
                # torchvision.transforms.RandomVerticalFlip(0.5),
                tricks.RandomErasing(mean=mean_rgb)
            ])
            img = trans(img)

        else:
            img = torchvision.transforms.Resize(197)(img)
    else:
        img = torchvision.transforms.Resize(197)(img)

    img = np.array(img, dtype=np.uint8)
    return transform(img)


def get_data_kfold(k=5):
    train_file_path = "Faces_in_the_Wild/train_relationships.csv"
    train_folders_path = "Faces_in_the_Wild/train/"

    all_images = glob(train_folders_path + "*/*/*.jpg")
    all_images = [x.replace('\\', '/') for x in all_images]

    families = set([x.split("/")[-3] for x in all_images])
    families = np.array(families)

    kf = KFold(n_splits=k, shuffle=False)
    for train_indexs, val_indexs in kf.split(families):
        val_families = families[val_indexs]

        val_images = []
        for val_family in val_families:
            val_images.extend([x for x in all_images if val_family in x])

        train_images = [x for x in all_images if x not in val_images]

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

        val = []
        for val_family in val_families:
            val.extend([x for x in relationships if val_family in x[0]])
        train = [x for x in relationships if x not in val]

        yield train, train_person_to_images_map, val, val_person_to_images_map


def get_data(val_famillies, extension=False,kinfacew = False):
    train_file_path = "Faces_in_the_Wild/train_relationships.csv"
    train_file_path_ext = 'KinFaceW-II/kfacew_2.csv'

    train_folders_path = "Faces_in_the_Wild/train/"


    all_images = glob(train_folders_path + "*/*/*.jpg")
    all_images = [x.replace('\\', '/') for x in all_images]

    # train_images, val_images = train_test_split(all_images, test_size=0.1)

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

    if extension:
        vgg_face = pd.read_csv('vgg_face.csv')
        pairs = vgg_face['img_pair'][vgg_face['is_related'] > 0.9].to_numpy()
        for pair in pairs:
            pp1, pp2 = pair.split('-')
            train.append((pp1, pp2))
            train_person_to_images_map[pp1] = ['Faces_in_the_Wild/test/' + pp1]
            train_person_to_images_map[pp2] = ['Faces_in_the_Wild/test/' + pp2]
    if kinfacew:
        relationships_ext = pd.read_csv(train_file_path_ext)
        relationships_ext = list(zip(relationships_ext.p1.values, relationships_ext.p2.values))

        train.extend(relationships_ext)
        for p1, p2 in relationships_ext:
            train_person_to_images_map[p1] = get_kinfacew_path(p1)
            train_person_to_images_map[p2] = get_kinfacew_path(p2)

    return train, train_person_to_images_map, val, val_person_to_images_map


def get_kinfacew_path(p):
    if p.startswith("fd"):
        return ["KinFaceW-II/images/father-dau/" + p]
    elif p.startswith('fs'):
        return ["KinFaceW-II/images/father-son/" + p]
    elif p.startswith('md'):
        return ['KinFaceW-II/images/mother-dau/' + p]
    elif p.startswith('ms'):
        return ['KinFaceW-II/images/mother-son/' + p]


class FaceDataSet(Dataset):
    def __init__(self, relations, label_images_map, kind, argument=False):
        self.relations = relations
        self.label_images_map = label_images_map
        self.kind = kind
        self.length = self.get_length()
        self.argument = argument
        # self.family_label_map, self.label_map = self.get_family_label_map()

    def get_family_label_map(self):
        family_label_map = defaultdict(list)
        for x in self.label_images_map:
            family = x.split("/")[0]
            family_label_map[family].append(x)

        label_map = defaultdict(list)
        for relation in self.relations:
            label_map[relation[0]].append(relation[1])

        return family_label_map, label_map

    def __getitem__(self, index):

        # should_same = self.same
        # self.same = not self.same

        should_same = index % 2 == 0
        if should_same:
            p1, p2 = self.relations[int(index / 2)]
            img1 = loader(choice(self.label_images_map[p1]), self.kind, self.argument)
            img2 = loader(choice(self.label_images_map[p2]), self.kind, self.argument)
            return img1, img2, torch.Tensor([1])
        else:
            # if random.uniform(0, 1) > 0.9:
            #     ii = 0
            #     while ii < 10:
            #         ii += 1
            #         fam = choice(list(self.family_label_map.keys()))
            #         if len(self.family_label_map[fam]) > 2:
            #             p1, p2 = sample(self.family_label_map[fam], 2)
            #             if p2 not in self.label_map[p1] and p1 not in self.label_map[p2]:
            #                 img1 = loader(choice(self.label_images_map[p1]), self.kind, self.argument)
            #                 img2 = loader(choice(self.label_images_map[p2]), self.kind, self.argument)
            #                 return img1, img2, torch.Tensor([0])
            while True:
                p1, p4 = sample(self.label_images_map.keys(), 2)
                # if 'fd' in p1 or 'fs' in p1 or 'md' in p1 or 'ms' in p1:
                #     continue
                # if 'fd' in p4 or 'fs' in p4 or 'md' in p4 or 'ms' in p4:
                #     continue
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


class FaceDataSet_V2(Dataset):
    def __init__(self, relations, label_images_map, kind, argument=False):
        self.relations = relations
        self.label_images_map = label_images_map
        self.kind = kind
        self.length = self.get_length()
        self.argument = argument
        self.family_label_map, self.label_map = self.get_family_label_map()

    def get_family_label_map(self):
        family_label_map = defaultdict(list)
        for x in self.label_images_map:
            family = x.split("/")[0]
            family_label_map[family].append(x)

        label_map = defaultdict(list)
        for relation in self.relations:
            label_map[relation[0]].append(relation[1])

        return family_label_map, label_map

    def __getitem__(self, index):
        p1, p2 = self.relations[int(index / 2)]
        img1 = loader(choice(self.label_images_map[p1]), self.kind, self.argument)
        img2 = loader(choice(self.label_images_map[p2]), self.kind, self.argument)
        family_id = p1.split("/")[0]
        labels = set(self.family_label_map[family_id])
        if len(labels) > 2 and random.uniform(0, 1) > 0.9:

            labels.remove(p1)
            labels.remove(p2)
            i = 0
            while i < 10:
                i += 1
                p3 = choice(list(labels))
                if (p1, p3) not in self.relations and (p3, p1) not in self.relations and (
                        p1, p2) not in self.relations and (p2, p1) not in self.relations:
                    img3 = loader(choice(self.label_images_map[p3]), self.kind, self.argument)
                    return img1, img2, img3, torch.Tensor([1])

        family_set = list(self.family_label_map.keys())
        family_set.remove(family_id)
        p3 = choice(self.family_label_map[choice(family_set)])
        img3 = loader(choice(self.label_images_map[p3]), self.kind, self.argument)
        return img1, img2, img3, torch.Tensor([1])

    def get_length(self):
        length = 0
        for key in self.label_images_map:
            length = length + len(self.label_images_map[key])
        return length

    def __len__(self):
        # if self.kind == 'train':
        #     return self.length
        return self.length

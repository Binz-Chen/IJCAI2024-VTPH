from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import copy

from model.clip_model.simple_tokenizer import SimpleTokenizer
import os
import numpy as np
import scipy.io as scio

from torch.utils.data import Dataset
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import hdf5storage


class BaseDataset(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 img_captions: dict,
                 is_train=True,
                 tokenizer=SimpleTokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 noise_ratio=0.,
                 noise_file="-",
                 dataset=None,
                 ):
        # Disrupt some correspondences of the training data according to the noise rate
        length = len(captions)
        self.t2i_index = np.arange(0, length)
        self.is_train = is_train
        if is_train:
            noise_length = int(noise_ratio * length)
            self._t2i_index = copy.deepcopy(self.t2i_index)
            if os.path.exists(noise_file):
                self.t2i_index = np.load(noise_file)
            else:
                idx_list = [i for i in range(length)]
                # Fixed seed=1, the seed can be changed to cover different cases
                np.random.seed(1)
                random.shuffle(idx_list)
                shuffle_index = self.t2i_index[idx_list[:noise_length]]
                np.random.seed(1)
                np.random.shuffle(shuffle_index)
                self.t2i_index[idx_list[:noise_length]] = shuffle_index

            self.corr_labels = np.ones(length, dtype="int")
            self.corr_labels[self._t2i_index != self.t2i_index] = 0

        self.captions = captions
        self.img_captions = img_captions
        self.indexs = indexs
        self.labels = labels
        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def load_image(self, index: int) -> torch.Tensor:
        image_path = self.indexs[index].strip()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def get_image_path(self, index: int) -> torch.Tensor:
        return self.indexs[index].strip()

    def get_text(self, index: int):
        return self.captions[index]

    def load_text(self, index: int):
        captions = self.captions[index]
        words = self.tokenizer.tokenize(captions)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)
        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask

    def load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def load_img_caption(self, index):
        img_captions = self.img_captions[index]
        words = self.tokenizer.tokenize(img_captions)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        img_caption = self.tokenizer.convert_tokens_to_ids(words)
        while len(img_caption) < self.maxWords:
            img_caption.append(0)
        img_caption = torch.tensor(img_caption)
        key_padding_mask = (img_caption == 0)
        return img_caption, key_padding_mask

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def get_images(self, index):
        images = []
        for i in index:
            images.append(self.load_image(i).unsqueeze(0))

        return torch.cat(images, dim=0)

    def get_img_captions(self, index):
        img_captions = []
        key_padding_masks = []
        for i in index:
            img_caption, key_padding_mask = self.load_img_caption(i)
            img_captions.append(img_caption.unsqueeze(0))
            key_padding_masks.append(key_padding_mask.unsqueeze(0))

        return torch.cat(img_captions, dim=0), torch.cat(key_padding_masks, dim=0)

    def get_texts(self, index):
        texts = []
        key_padding_masks = []
        for i in index:
            text, key_padding_mask = self.load_img_caption(i)
            texts.append(text.unsqueeze(0))
            key_padding_masks.append(key_padding_mask.unsqueeze(0))

        return torch.cat(texts, dim=0), torch.cat(key_padding_masks, dim=0)

    def __getitem__(self, index):
        image = self.load_image(index)
        caption, key_padding_mask = self.load_text(self.t2i_index[index])
        label = self.load_label(index)
        img_caption, key_padding_mask_cap = self.load_img_caption(index)
        sample = {'img_pth': self.indexs[index].strip(), 'text': self.captions[index], 'img_caption': self.img_captions[index]}
        return image, caption, key_padding_mask, label, index, img_caption, key_padding_mask_cap, sample


def split_data(captions, indexs, labels, img_captions, query_num, train_num, seed=1):
    np.random.seed(seed)  # fixed to 1 for all experiments.

    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]
    query_img_captions = img_captions[query_index]

    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]
    train_img_captions = img_captions[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]
    retrieval_img_captions = img_captions[retrieval_index]

    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)
    split_img_captions = (query_img_captions, train_img_captions, retrieval_img_captions)

    return split_indexs, split_captions, split_labels, split_img_captions


def generate_dataset(captionFile: str,
                     indexFile: str,
                     labelFile: str,
                     img_captionFile: str,
                     maxWords=32,
                     imageResolution=224,
                     query_num=2000,
                     train_num=10000,
                     seed=1,
                     noise_ratio=0.,
                     dataset=None,
                     ):
    try:
        captions = scio.loadmat(captionFile)["caption"]
    except:
        captions = hdf5storage.loadmat(captionFile)["caption"]
    indexs = scio.loadmat(indexFile)["index"]
    labels = scio.loadmat(labelFile)["label"]
    img_captions = scio.loadmat(img_captionFile)["image_caption"]

    split_indexs, split_captions, split_labels, split_img_captions = split_data(captions, indexs, labels, img_captions,
                                                                                query_num=query_num,
                                                                                train_num=train_num, seed=seed)
    query_data = BaseDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             img_captions=split_img_captions[0],
                             maxWords=maxWords, imageResolution=imageResolution, is_train=False)
    train_data = BaseDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             img_captions=split_img_captions[1],
                             maxWords=maxWords, imageResolution=imageResolution, noise_ratio=noise_ratio,
                             dataset=dataset)
    retrieval_data = BaseDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 img_captions=split_img_captions[2],
                                 maxWords=maxWords, imageResolution=imageResolution, is_train=False)

    return train_data, query_data, retrieval_data

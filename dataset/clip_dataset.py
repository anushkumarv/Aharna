import clip

import pandas as pd
from PIL import Image
import requests
import re

import torch
import torch.utils.data as data

import os 


class TrainDatasetOnline(data.Dataset):

    def __init__(self, config: dict, transforms: None) -> None:
        super().__init__()
        self.config = config
        self.transforms = transforms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(config.get('clip_backend'), device=self.device)
        self.train_image_emb = None 
        self.train_text_emb = None

    def __getitem__(self, index):
        return self.train_image_emb[index], self.train_text_emb[index]

    def __len__(self):
        return self.train_image_emb.shape[0]

    def _download_image(self, img_id):
        try:
            image = Image.open(requests.get(self.config.get('url_root') + img_id + '.jpg', stream=True).raw)
        except Exception as e:
            return None
        if self.transform is not None:
            image = self.transform(image)

        return image

    def _get_image_embedding(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def _preprocess_text(self, text):
        return re.sub(r'[^A-Za-z0-9 ]+', '', text).lower()

    def _get_text_embeddings(self, text1, text2, text3):
        text = self._preprocess_text(text1) + self.config.get('text_sep') +\
               self._preprocess_text(text2) + self.config.get('text_sep') +\
               self._preprocess_text(text3)

        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(text)

        return text_emb

    def _prepare_embeddings(self):
        src_file = os.path.join(self.config.get('data_root') + self.config.get('train_csv'))
        df = pd.read_csv(src_file)




import clip

import pandas as pd
from PIL import Image
import requests
import re

import torch
import torch.utils.data as data

import os 


class TrainClipDatasetOnline(data.Dataset):

    def __init__(self, config: dict, transforms=None) -> None:
        super().__init__()
        self.config = config
        self.transforms = transforms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(config.get('clip_backend'), device=self.device)
        self.train_compress_net_emb = None 
        self.train_query_net_emb = None
        self._prepare_embeddings()

    def __getitem__(self, index):
        return self.train_compress_net_emb[index], self.train_query_net_emb[index]

    def __len__(self):
        return self.train_image_emb.shape[0]

    def _download_image(self, img_id):
        try:
            image = Image.open(requests.get(self.config.get('url_root') + img_id + '.jpg', stream=True).raw)
        except Exception as e:
            raise Exception
        if self.transform is not None:
            image = self.transform(image)

        return image

    def _get_image_embedding(self, src_image, tgt_image, ntgt_image):
        src_image = self.preprocess(src_image).unsqueeze(0).to(self.device)
        tgt_image = self.preprocess(tgt_image).unsqueeze(0).to(self.device)
        ntgt_image = self.preprocess(ntgt_image).unsqueeze(0).to(self.device)
        all_image = torch.vstack((src_image,tgt_image,ntgt_image))
        with torch.no_grad():
            all_image_features = self.model.encode_image(all_image)
        return all_image_features

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

    def _stack_embeddings(self, all_img_emb, text_em):
        src_img_em, tgt_ntgt_img_emb = torch.split(all_img_emb, [1,2])
        tgt_ntgt_img_emb = tgt_ntgt_img_emb.unsqueeze(0)
        query_em = torch.cat((src_img_em, text_em), dim=1).unsqueeze(0)

        if self.train_compress_net_emb:
            self.train_compress_net_emb = torch.vstack((self.train_compress_net_emb,tgt_ntgt_img_emb))
        else:
            self.train_compress_net_emb = tgt_ntgt_img_emb

        if self.train_query_net_emb:
            self.train_query_net_emb = torch.vstack((self.train_query_net_emb, query_em))
        else:
            self.train_query_net_emb = query_em

    def _prepare_embeddings(self):
        src_file = os.path.join(self.config.get('data_root') + self.config.get('train_csv'))
        df = pd.read_csv(src_file)
        for _, row in df.iterrows:
            try:
                all_img_emb = self._get_image_embedding(self._download_image(row['Source Image ID']), 
                                        self._download_image(row['Target Image ID']),
                                        self._download_image(row['Non-Target Image ID']))
                txt_em = self._get_text_embeddings(row['Feedback 1'], row['Feedback 2'], row['Feedback 3'])
                self._stack_embeddings(all_img_emb, txt_em)
            except Exception as e:
                continue





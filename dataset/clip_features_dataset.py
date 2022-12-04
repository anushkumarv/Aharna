import clip

import json
import pandas as pd
from PIL import Image
import requests
import re
from tqdm import tqdm

import torch
import torch.utils.data as data

import os 


class PrdFeedbackClipBkdDataset(data.Dataset):

    def __init__(self, config: dict, transforms=None) -> None:
        super().__init__()
        self.config = config
        self.transforms = transforms
        self.device = "cpu"
        self.model, self.preprocess = clip.load(config.get('clip_backend'), device=self.device)
        self.train_tgt_img_emb = None
        self.train_ntgt_img_emb = None
        self.train_qnet_emb = None
        if os.path.exists(config.get('train_tgt_img_emb')) and os.path.exists(config.get('train_ntgt_img_emb')) and os.path.exists(config.get('train_qnet_emb')):
            self.train_tgt_img_emb = torch.load(config.get('train_tgt_img_emb')).to(torch.float32)
            self.train_ntgt_img_emb = torch.load(config.get('train_ntgt_img_emb')).to(torch.float32)
            self.train_qnet_emb = torch.load(config.get('train_qnet_emb')).to(torch.float32)
        else:
            self._prepare_embeddings()
            torch.save(self.train_tgt_img_emb, config.get('train_tgt_img_emb'))
            torch.save(self.train_ntgt_img_emb, config.get('train_ntgt_img_emb'))
            torch.save(self.train_qnet_emb, config.get('train_qnet_emb'))

    def __getitem__(self, index):
        return self.train_qnet_emb[index], self.train_tgt_img_emb[index], self.train_ntgt_img_emb[index]

    def __len__(self):
        return len(self.train_qnet_emb)

    def _read_img(self, img_id):
        try:
            image = Image.open(os.path.join(self.config.get('train_img'), img_id + '.jpg'))
        except Exception as e:
            raise Exception
        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def _download_image(self, img_id):
        try:
            image = Image.open(requests.get(self.config.get('url_root') + img_id + '.jpg', stream=True).raw)
        except Exception as e:
            raise Exception
        if self.transforms is not None:
            image = self.transforms(image)

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
        src_img_em, tgt_img_emb, ntgt_img_emb = torch.split(all_img_emb, [1,1,1])
        query_em = torch.cat((src_img_em, text_em), dim=1)

        if self.train_qnet_emb is not None:
            self.train_qnet_emb = torch.vstack((self.train_qnet_emb, query_em))
        else:
            self.train_qnet_emb = query_em

        if self.train_tgt_img_emb is not None:
            self.train_tgt_img_emb = torch.vstack((self.train_tgt_img_emb, tgt_img_emb))
        else:
            self.train_tgt_img_emb = tgt_img_emb

        if self.train_ntgt_img_emb is not None:
            self.train_ntgt_img_emb = torch.vstack((self.train_ntgt_img_emb, ntgt_img_emb))
        else:
            self.train_ntgt_img_emb = ntgt_img_emb

    def _prepare_embeddings(self):
        src_file = os.path.join(self.config.get('data_root'), self.config.get('train_csv'))
        df = pd.read_csv(src_file)
        count = 0
        for _, row in tqdm(df.iterrows()):
            try:
                if self.config.get('cap_datapoints') and count == self.config.get('max_datapoints'):
                    break
                if self.config.get('read_from_folder'):
                    all_img_emb = self._get_image_embedding(self._read_img(row['Source Image ID']), 
                                            self._read_img(row['Target Image ID']),
                                            self._read_img(row['Non-Target Image ID']))
                else:
                    all_img_emb = self._get_image_embedding(self._download_image(row['Source Image ID']), 
                                            self._download_image(row['Target Image ID']),
                                            self._download_image(row['Non-Target Image ID']))
                txt_em = self._get_text_embeddings(row['Feedback 1'], row['Feedback 2'], row['Feedback 3'])
                self._stack_embeddings(all_img_emb, txt_em)
                count += 1
            except Exception as e:
                continue

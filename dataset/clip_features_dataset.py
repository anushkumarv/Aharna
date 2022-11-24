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
        self.train_compress_net_emb = None 
        self.train_query_net_emb = None
        if os.path.exists(config.get('train_cnet_em')) and os.path.exists(config.get('train_qnet_em')):
            self.train_compress_net_emb = torch.load(config.get('train_cnet_em'))
            self.train_query_net_emb = torch.load(config.get('train_qnet_em'))
        else:
            self._prepare_embeddings()
            torch.save(self.train_compress_net_emb, config.get('train_cnet_em'))
            torch.save(self.train_query_net_emb, config.get('train_qnet_em'))

    def __getitem__(self, index):
        return self.train_compress_net_emb[index], self.train_query_net_emb[index]

    def __len__(self):
        return len(self.train_compress_net_emb)

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
        src_img_em, tgt_ntgt_img_emb = torch.split(all_img_emb, [1,2])
        tgt_ntgt_img_emb = tgt_ntgt_img_emb.unsqueeze(0)
        query_em = torch.cat((src_img_em, text_em), dim=1).unsqueeze(0)

        if self.train_compress_net_emb is not None:
            self.train_compress_net_emb = torch.vstack((self.train_compress_net_emb,tgt_ntgt_img_emb))
        else:
            self.train_compress_net_emb = tgt_ntgt_img_emb

        if self.train_query_net_emb is not None:
            self.train_query_net_emb = torch.vstack((self.train_query_net_emb, query_em))
        else:
            self.train_query_net_emb = query_em

    def _prepare_embeddings(self):
        src_file = os.path.join(self.config.get('data_root'), self.config.get('train_csv'))
        df = pd.read_csv(src_file)
        count = 0
        for _, row in tqdm(df.iterrows()):
            try:
                if self.config.get('cap_datapoints') and count == self.config.get('max_datapoints'):
                    break
                all_img_emb = self._get_image_embedding(self._download_image(row['Source Image ID']), 
                                        self._download_image(row['Target Image ID']),
                                        self._download_image(row['Non-Target Image ID']))
                txt_em = self._get_text_embeddings(row['Feedback 1'], row['Feedback 2'], row['Feedback 3'])
                self._stack_embeddings(all_img_emb, txt_em)
                count += 1
            except Exception as e:
                continue


class PrdFeedbackClipBkdDevDataset(data.Dataset):
    def __init__(self, config: dict, transforms = None):
        super().__init__()
        self.config = config
        self.transforms = transforms
        self.device = "cpu"
        self.model, self.preprocess = clip.load(config.get('clip_backend'), device=self.device)
        self.dev_query_net_emb = None
        if os.path.exists(config.get('dev_qnet_em')):
            self.dev_query_net_emb = torch.load(config.get('dev_qnet_em'))
        else:
            self._prepare_embeddings()
            torch.save(self.dev_query_net_emb, config.get('dev_qnet_em'))

    def __getitem__(self, index):
        return self.dev_query_net_emb[index]

    def __len__(self):
        return len(self.dev_query_net_emb)

    def _download_image(self, img_id):
        try:
            image = Image.open(requests.get(self.config.get('url_root') + img_id + '.jpg', stream=True).raw)
        except Exception as e:
            raise Exception
        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def _get_image_embedding(self, src_image):
        src_image = self.preprocess(src_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            src_image_features = self.model.encode_image(src_image)
        return src_image_features

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

    def _stack_embeddings(self, src_img_em, text_em):
        query_em = torch.cat((src_img_em, text_em), dim=1).unsqueeze(0)

        if self.dev_query_net_emb is not None:
            self.dev_query_net_emb = torch.vstack((self.dev_query_net_emb, query_em))
        else:
            self.dev_query_net_emb = query_em

    def _prepare_embeddings(self):
        src_file = os.path.join(self.config.get('data_root'), self.config.get('dev_jsonl'))
        with open(src_file, 'r') as jsonl_file:
            json_str = jsonl_file.read()

        data = [json.loads(json_item) for json_item in json_str.splitlines()]
        for item in tqdm(data):
            try:
                src_img_em = self._get_image_embedding(self._download_image(item['source_pid']))
                text_em = self._get_text_embeddings(item['feedback1'],item['feedback2'],item['feedback3'])
                self._stack_embeddings(src_img_em, text_em)
            except Exception as e:
                continue    


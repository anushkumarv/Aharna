import clip
import json
from nn_modules import clip_bknd_modules
import os
from PIL import Image
import re
import requests
import torch
import torch.nn as nn
from tqdm import tqdm


class QryInf():
    def __init__(self, config, transforms=None):
        self.config  = config
        self.transforms = transforms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = list()
        self.clip_model, self.clip_preprocess = clip.load(config.get('clip_backend'), device=self.device)
        if self.config.get('use_clip_aprch_inf'):
            if self.config.get('clip_backend') == 'ViT-B/32':
                self.cnet = clip_bknd_modules.ClipBackendB32CompressNet()
                self.qnet = clip_bknd_modules.ClipBackendB32QueryNet()
                self.clip_model_fsize = 512
            else:
                self.cnet = clip_bknd_modules.ClipBackendB32CompressNet()
                self.qnet = clip_bknd_modules.ClipBackendL14QueryNet()
                self.clip_model_fsize = 768

            
            self.cnet.load_state_dict(torch.load(self.config.get('clip_bknd_cnet_model_path')))
            self.qnet.load_state_dict(torch.load(self.config.get('clip_bknd_qnet_model_path')))
            self.cnet.eval()
            self.qnet.eval()
            self.cnet.to(self.device)
            self.qnet.to(self.device)


    def _download_and_preprocess_image(self, img_id):
        try:
            image = Image.open(requests.get(self.config.get('url_root') + img_id + '.jpg', stream=True).raw)
        except Exception as e:
            return None
        if self.transforms is not None:
            image = self.transforms(image)

        return self.clip_preprocess(image).unsqueeze(0).to(self.device)

    def _get_image_embedding(self, image):
        if image is None:
            return None
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
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

    def _get_preds(self, can_img_em, query_em):
        cnet_y = self.cnet(can_img_em)
        qnet_y = self.qnet(query_em)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        preds = cos(cnet_y, qnet_y)
        return preds

    def _write_res(self, data):
        res_file = self.config.get('dev_phase_res_file')
        with open(res_file, 'w') as f:
            for item in data:
                f.write(str(item) + '\n')

    def infer_data(self):
        src_file = os.path.join(self.config.get('data_root'), self.config.get('dev_jsonl'))
        with open(src_file, 'r') as jsonl_file:
            json_str = jsonl_file.read()

        data = [json.loads(json_item) for json_item in json_str.splitlines()]
        print("starting predictions")
        for item in tqdm(data):
            src_img_em = self._get_image_embedding(self._download_and_preprocess_image(item['source_pid']))
            src_img_em = src_img_em if src_img_em is not None else torch.zeros(1,self.clip_model_fsize).to(self.device)
            can_img_em = None
            for candidate_imgs in item['candidates']:
                can_img = self._get_image_embedding(self._download_and_preprocess_image(candidate_imgs['candidate_pid']))
                can_img  = can_img if can_img is not None else torch.zeros(1,self.clip_model_fsize).to(self.device)
                if can_img_em is not None:
                    can_img_em = torch.vstack((can_img_em, can_img))
                else:
                    can_img_em  = can_img
            can_img_em = can_img_em.to(torch.float32)
            text_em = self._get_text_embeddings(item['feedback1'],item['feedback2'],item['feedback3'])
            query_em = torch.cat((src_img_em, text_em), dim=1)
            text_em = text_em.to(torch.float32)
            query_em = query_em.to(torch.float32)
            preds = self._get_preds(can_img_em, query_em)
            for i in range(len(item['candidates'])):
                item['candidates'][i]["score"] = round(preds[i].data.item(),2)
        print("writing results to file")
        self._write_res(data)


            

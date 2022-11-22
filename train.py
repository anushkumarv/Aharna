from config import config
from dataset import clip_features_dataset
from nn_modules import clip_bknd_modules

import torch




device = "cuda" if torch.cuda.is_available() else "cpu"

train_data_loader = torch.utils.data.DataLoader(
        clip_features_dataset.TrainClipDatasetOnline(config=config),
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers'),
        shuffle=False,
        pin_memory=True,
    )



from config import config
from dataset import clip_features_dataset
from joint_training import joint_trainning
from nn_modules import clip_bknd_modules
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

torch.manual_seed(config.get('random_seed'))


device = "cuda" if torch.cuda.is_available() else "cpu"

def display_loss_curve(history, title, save_img_file, include_val_loss = False):
    loss = history['loss']
    epochs = list(range(1, len(loss) + 1))
    plt.plot(epochs, loss, 'b', label='Training loss')
    if include_val_loss:
        val_loss = history['val_loss']
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(save_img_file)

print("preparing dataset and dataloader")


prd_fdbk_dataset = clip_features_dataset.PrdFeedbackClipBkdDataset(config=config)
train_data_loader = None
val_data_loader = None
if config.get('split_train_into_val'):
    indices = list(range(len(prd_fdbk_dataset)))
    split = int(np.floor(config.get('validation_split') * len(prd_fdbk_dataset)))
    if config.get('shuffle_data') :
        np.random.seed(config.get('random_seed'))
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_data_loader = torch.utils.data.DataLoader(
            prd_fdbk_dataset,
            batch_size=config.get('batch_size'),
            num_workers=config.get('num_workers'),
            shuffle=config.get('shuffle_data'),
            pin_memory=True,
            sampler=train_sampler,
        )
    val_data_loader = torch.utils.data.DataLoader(
            prd_fdbk_dataset,
            batch_size=config.get('batch_size'),
            num_workers=config.get('num_workers'),
            shuffle=config.get('shuffle_data'),
            pin_memory=True,
            sampler=valid_sampler,
        )

else:
    train_data_loader = torch.utils.data.DataLoader(
            prd_fdbk_dataset,
            batch_size=config.get('batch_size'),
            num_workers=config.get('num_workers'),
            shuffle=False,
            pin_memory=True,
        )

print("starting training")
# loss = nn.CrossEntropyLoss()
loss = nn.CosineEmbeddingLoss()
cnet_model = clip_bknd_modules.ClipBackendB32CompressNet()
qnet_model = clip_bknd_modules.ClipBackendB32QueryNet()
cnet_model.to(device=device)
qnet_model.to(device=device)
opt = torch.optim.Adam(list(cnet_model.parameters()) + list(qnet_model.parameters()), lr = config.get('learning_rate'))

history = joint_trainning(cnet_model, qnet_model, opt, None, loss, train_data_loader, val_data_loader, config.get('epochs'), device)

torch.save(cnet_model.state_dict(), config.get('clip_bknd_cnet_model_path'))
torch.save(qnet_model.state_dict(), config.get('clip_bknd_qnet_model_path'))

display_loss_curve(history, "cnet and qnet loss", config.get('clip_crs_em_loss_img'), include_val_loss=config.get('split_train_into_val'))


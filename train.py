from config import config
from dataset import clip_features_dataset
from joint_training import joint_trainning
from nn_modules import clip_bknd_modules
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn

torch.manual_seed(config.get('random_seed'))


device = "cuda" if torch.cuda.is_available() else "cpu"

def display_loss_curve(history, title, save_img_file):
    loss = history['loss']
    # val_loss = history['val_loss']
    epochs = list(range(1, len(loss) + 1))
    plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(save_img_file)

print("preparing dataset and dataloader")
train_data_loader = torch.utils.data.DataLoader(
        clip_features_dataset.TrainClipDatasetOnline(config=config),
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers'),
        shuffle=False,
        pin_memory=True,
    )

print("starting training")
loss = nn.CrossEntropyLoss()
cnet_model = clip_bknd_modules.ClipBackendB32CompressNet()
qnet_model = clip_bknd_modules.ClipBackendB32QueryNet()
cnet_model.to(device=device)
qnet_model.to(device=device)
opt = torch.optim.Adam(list(cnet_model.parameters()) + list(qnet_model.parameters()), lr = config.get('learning_rate'))

history = joint_trainning(cnet_model, qnet_model, opt, loss, train_data_loader, None, 10, device)

display_loss_curve(history, "cnet and qnet loss", config.get('clip_crs_em_loss_img'))


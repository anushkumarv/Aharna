import torch
from torch import nn

class ClipBackendB32QueryNet(nn.Module):
    def __init__(self):
        super(ClipBackendB32QueryNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.query_net = nn.Sequential(nn.Linear(1024, 512, device=self.device),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.7), 
                                        nn.Linear(512, 128, device=self.device),
                                        nn.ReLU())

    def forward(self, x):
        # return nn.functional.normalize(self.query_net(x))
        return self.query_net(x)


class ClipBackendB32CompressNet(nn.Module):
    def __init__(self):
        super(ClipBackendB32CompressNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compress_net = nn.Sequential(nn.Linear(512, 128, device=self.device),
                                            nn.ReLU())

    def forward(self, x):
        # return nn.functional.normalize(self.compress_net(x))
        return self.compress_net(x)


class ClipBackendL14QueryNet(nn.Module):
    def __init__(self):
        super(ClipBackendL14QueryNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.query_net = nn.Sequential(nn.Linear(1536, 512, device=self.device),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.7), 
                                        nn.Linear(512, 128, device=self.device),
                                        nn.ReLU())

    def forward(self, x):
        return nn.functional.normalize(self.query_net(x))


class ClipBackendL14CompressNet(nn.Module):
    def __init__(self):
        super(ClipBackendL14CompressNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compress_net = nn.Sequential(nn.Linear(768, 128, device=self.device),
                                            nn.ReLU())

    def forward(self, x):
        return nn.functional.normalize(self.compress_net(x))

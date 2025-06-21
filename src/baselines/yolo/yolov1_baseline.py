import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, img_size=448):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        # ------- feature extractor -------
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                       # /4
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                       # /8
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                       # /16
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                       # /32
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)                        # /64  → 448/64 = 7  (pont a YOLOv1-hez)
        )

        # ------- számoljuk ki a flatten dim-et -------
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)  # dummy input
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        # ------- teljesen összekötő rétegek -------
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + 5 * B))
        )

    def forward(self,x):
        x = self.features(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.C + 5*self.B)

        # aktivációk
        x[..., self.C:self.C+4*self.B] = torch.sigmoid(x[..., self.C:self.C+4*self.B])
        x[..., self.C+4::5]           = torch.sigmoid(x[..., self.C+4::5])         # conf
        # osztályokat majd a loss-ban softmaxoljuk
        return x

class MediumYOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=5, img_size=448):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        # ⬆️  nagyobb csatornaszámok + több réteg
        self.features = nn.Sequential(
            nn.Conv2d(3,   64, 7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,  128, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, 1),            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                # itt 448/32 = 14 térbeli méret
            nn.Conv2d(512,1024,3, padding=1),  nn.LeakyReLU(0.1),
            nn.Conv2d(1024,512,1),             nn.LeakyReLU(0.1),
            nn.Conv2d(512,1024,3, padding=1),  nn.LeakyReLU(0.1),
            nn.Conv2d(1024,512,1),             nn.LeakyReLU(0.1),
            nn.Conv2d(512,1024,3, padding=1),  nn.LeakyReLU(0.1),
        )

        # --------- dinamikusan számoljuk ki a flatten méretet ----------
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_dim = self.features(dummy).numel()

        # ---------------- FC fej -------------------
        self.fc = nn.Sequential(
            nn.Flatten(),                               # -> [B, flat_dim]
            nn.Linear(flat_dim, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + 5 * B))
        )

    def forward(self, x):
        x = self.features(x)                            # [B,1024,14,14]
        x = self.fc(x)                                  # [B, S*S*(C+5B)]
        x = x.view(-1, self.S, self.S, self.C + 5*self.B)
        return x
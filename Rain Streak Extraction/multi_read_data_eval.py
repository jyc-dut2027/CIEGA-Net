import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
# from skimage import io
# from skimage import color
# import cv2
import config


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, pathlistfile, img_dir, N=5):
        self.img_dir = img_dir
        self.N = N
        self.pathlist = self.loadpath(pathlistfile)
        self.count = len(self.pathlist)

    def loadpath(self, pathlistfile):
        with open(pathlistfile) as fp:
            return fp.read().splitlines()

    def __getitem__(self, idx):
        # 滑窗：以 idx 为中心，取前后各 N//2
        center = idx
        half = self.N // 2
        indices = list(range(center - half, center + half + 1))
        # 边界保护（可选：不足补空/丢弃，建议在外层循环滑窗区间时已经避免越界）
        imgs = []
        for i in indices:
            img_path = os.path.join(self.img_dir, self.pathlist[i])
            img = plt.imread(img_path)[0:config.h, 0:config.w] / 255.0
            imgs.append(img)
        imgs = np.stack(imgs)   # [N, H, W, C]
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # [N, C, H, W]
        return torch.from_numpy(imgs).float(), self.pathlist[center]


    def __len__(self):
        return self.count


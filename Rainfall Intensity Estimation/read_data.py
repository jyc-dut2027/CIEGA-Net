import hashlib
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
RE_NEW = re.compile(r"^(\d+)_([0-9]+)_([0-9]+(?:\.[0-9]+)?)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
RE_OLD = re.compile(r"^(\d+)_([0-9]+(?:\.[0-9]+)?)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)


def parse_filename(fname: str):
    """从图片文件名解析序号、事件编号和雨强。"""
    base = os.path.basename(fname)

    match = RE_NEW.match(base)
    if match:
        seq = int(match.group(1))
        event_id = int(match.group(2))
        intensity = float(match.group(3))
        return seq, event_id, intensity

    match = RE_OLD.match(base)
    if match:
        seq = int(match.group(1))
        intensity = float(match.group(2))
        return seq, -1, intensity

    return None


def collect_image_files(root_dir: str, recursive: bool = False) -> List[str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    files = []
    if recursive:
        for cur_root, _, names in os.walk(root_dir):
            for fname in names:
                if fname.lower().endswith(IMG_EXTS) and parse_filename(fname) is not None:
                    files.append(os.path.join(cur_root, fname))
    else:
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(IMG_EXTS) and parse_filename(fname) is not None:
                files.append(os.path.join(root_dir, fname))

    files = sorted(files)
    if not files:
        raise ValueError(f"没有找到符合命名规则的图片: {root_dir}")
    return files


def split_train_val(file_paths: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"VAL_RATIO 必须在 [0, 1) 范围内，当前为 {val_ratio}")

    files = list(file_paths)
    random.Random(seed).shuffle(files)

    val_size = int(len(files) * val_ratio)
    val_files = files[:val_size]
    train_files = files[val_size:]

    if not train_files:
        raise ValueError("划分后训练集为空，请减小 VAL_RATIO。")
    return train_files, val_files


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_signature(image_paths: List[str]) -> str:
    digest = hashlib.md5()
    for path in sorted(image_paths):
        try:
            stat = os.stat(path)
            item = f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}\n"
        except FileNotFoundError:
            item = f"{os.path.abspath(path)}|missing\n"
        digest.update(item.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _sobel_pil(gray_img: Image.Image) -> Image.Image:
    arr = np.asarray(gray_img, dtype=np.uint8)
    grad_x = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad = np.clip(grad, 0, 255).astype(np.uint8)
    return Image.fromarray(grad, mode="L")


def prepare_pil_image(img: Image.Image, in_channels: int) -> Image.Image:
    if in_channels == 1:
        img = img.convert("YCbCr").split()[0] if config.USE_RGB_TO_Y else img.convert("L")
        return _sobel_pil(img) if config.USE_SOBEL_EDGE else img

    if in_channels == 3:
        img = img.convert("RGB")
        return _sobel_pil(img.convert("L")).convert("RGB") if config.USE_SOBEL_EDGE else img

    raise ValueError(f"INPUT_CHANNELS 只支持 1 或 3，当前为 {in_channels}")


def _resize_resample(in_channels: int):
    if in_channels == 1 and config.BINARY_NEAREST_RESIZE:
        return Image.NEAREST
    return Image.BILINEAR


def _compute_mean_std_streaming(
    image_paths: List[str],
    in_channels: int,
    input_size: Tuple[int, int],
    max_images: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    paths = list(image_paths)
    if max_images is not None and len(paths) > max_images:
        paths = random.Random(config.SPLIT_RANDOM_SEED).sample(paths, max_images)

    if in_channels == 1:
        sum_v = 0.0
        sum2_v = 0.0
    elif in_channels == 3:
        sum_v = np.zeros(3, dtype=np.float64)
        sum2_v = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError(f"INPUT_CHANNELS 只支持 1 或 3，当前为 {in_channels}")

    count = 0
    height, width = input_size
    resample = _resize_resample(in_channels)

    for path in tqdm(paths, desc="[Norm] 统计 mean/std", colour="green"):
        with Image.open(path) as img:
            img = prepare_pil_image(img, in_channels).resize((width, height), resample)
            arr = np.asarray(img, dtype=np.float32) / 255.0

        if in_channels == 1:
            values = arr.reshape(-1)
            sum_v += float(values.sum())
            sum2_v += float((values * values).sum())
            count += int(values.size)
        else:
            values = arr.reshape(-1, 3)
            sum_v += values.sum(axis=0)
            sum2_v += (values * values).sum(axis=0)
            count += int(values.shape[0])

    if count == 0:
        raise ValueError("mean/std 统计失败：没有有效像素。")

    mean = sum_v / count
    var = np.maximum(sum2_v / count - mean * mean, 1e-8)
    std = np.sqrt(var)

    if in_channels == 1:
        return [float(mean)], [float(std)]
    return [float(x) for x in mean.tolist()], [float(x) for x in std.tolist()]


def load_norm_stats_file(norm_file: str) -> Tuple[Optional[List[float]], Optional[List[float]], str]:
    if config.PER_IMAGE_NORMALIZE:
        print("[Norm] 使用单图标准化，不依赖全局 mean/std。")
        return None, None, norm_file

    if not os.path.isfile(norm_file):
        raise FileNotFoundError(f"找不到归一化统计文件: {norm_file}")

    with open(norm_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mean = data.get("mean")
    std = data.get("std")
    if mean is None or std is None:
        raise ValueError(f"归一化统计文件缺少 mean/std 字段: {norm_file}")

    print(f"[Norm] 使用: {norm_file}")
    print(f"[Norm] mean={mean}, std={std}")
    return mean, std, norm_file


def load_or_create_norm_stats(
    norm_file: str,
    stat_files: List[str],
    in_channels: int,
) -> Tuple[Optional[List[float]], Optional[List[float]], str]:
    _ensure_parent(norm_file)
    signature = _file_signature(stat_files)

    if config.PER_IMAGE_NORMALIZE:
        meta = {
            "mean": None,
            "std": None,
            "mode": "per_image_normalize",
            "input_channels": in_channels,
            "input_size": list(config.INPUT_SIZE),
            "file_count": len(stat_files),
            "file_signature": signature,
        }
        with open(norm_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[Norm] 使用单图标准化，记录写入: {norm_file}")
        return None, None, norm_file

    if in_channels == 1:
        if config.BINARY_NORM_POLICY == "skip":
            print("[Norm] 不使用 Normalize。")
            return None, None, norm_file
        if config.BINARY_NORM_POLICY == "fixed":
            mean, std = config.BINARY_FIXED_MEAN_STD
            with open(norm_file, "w", encoding="utf-8") as f:
                json.dump({"mean": list(mean), "std": list(std), "mode": "fixed"}, f, ensure_ascii=False, indent=2)
            return list(mean), list(std), norm_file
        if config.BINARY_NORM_POLICY != "compute":
            raise ValueError(f"未知 BINARY_NORM_POLICY: {config.BINARY_NORM_POLICY}")

    if config.NORM_USE_CACHE and not config.FORCE_REBUILD_NORM and os.path.isfile(norm_file):
        with open(norm_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("file_signature") == signature and data.get("input_channels") == in_channels:
            mean, std = data.get("mean"), data.get("std")
            if mean is not None and std is not None:
                print(f"[Norm] 使用缓存: {norm_file}")
                print(f"[Norm] mean={mean}, std={std}")
                return mean, std, norm_file

    mean, std = _compute_mean_std_streaming(
        image_paths=stat_files,
        in_channels=in_channels,
        input_size=config.INPUT_SIZE,
        max_images=config.MAX_STAT_IMAGES,
    )
    meta = {
        "mean": mean,
        "std": std,
        "mode": "dataset_normalize",
        "input_channels": in_channels,
        "input_size": list(config.INPUT_SIZE),
        "file_count": len(stat_files),
        "file_signature": signature,
    }
    with open(norm_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Norm] 已写入: {norm_file}")
    print(f"[Norm] mean={mean}, std={std}")
    return mean, std, norm_file


def build_transform(
    mean: Optional[List[float]],
    std: Optional[List[float]],
    in_channels: int,
    is_train: bool = False,
):
    interpolation = (
        transforms.InterpolationMode.NEAREST
        if in_channels == 1 and config.BINARY_NEAREST_RESIZE
        else transforms.InterpolationMode.BILINEAR
    )

    transform_list = [
        transforms.Lambda(lambda img: prepare_pil_image(img, in_channels)),
        transforms.Resize(config.INPUT_SIZE, interpolation=interpolation),
    ]

    if is_train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
        ])

    transform_list.append(transforms.ToTensor())

    if config.PER_IMAGE_NORMALIZE:
        transform_list.append(transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-6)))
    elif mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


class RainDataset(Dataset):
    def __init__(self, file_paths: List[str], transform=None, return_meta: bool = False):
        self.file_paths = list(file_paths)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        fname = os.path.basename(path)
        info = parse_filename(fname)
        if info is None:
            raise ValueError(f"无法从文件名解析信息: {fname}")

        seq, event_id, intensity = info
        with Image.open(path) as img:
            img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.return_meta:
            return img, int(seq), int(event_id), float(intensity), fname
        return img, float(intensity)


def make_loader(dataset: Dataset, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )


def build_train_val_loaders(root_dir: Optional[str] = None) -> Dict[str, object]:
    root_dir = config.train_dir if root_dir is None else root_dir
    all_files = collect_image_files(root_dir)

    if config.USE_VALIDATION:
        train_files, val_files = split_train_val(
            all_files,
            val_ratio=config.VAL_RATIO,
            seed=config.SPLIT_RANDOM_SEED,
        )
    else:
        train_files = all_files
        val_files = []

    if config.NORM_SCOPE == "train_only":
        stat_files = train_files
    elif config.NORM_SCOPE == "all_in_root":
        stat_files = all_files
    else:
        raise ValueError(f"未知 NORM_SCOPE: {config.NORM_SCOPE}")

    mean, std, norm_path = load_or_create_norm_stats(
        norm_file=config.NORM_STATS_FILE,
        stat_files=stat_files,
        in_channels=config.INPUT_CHANNELS,
    )

    train_transform = build_transform(mean, std, config.INPUT_CHANNELS, is_train=True)
    eval_transform = build_transform(mean, std, config.INPUT_CHANNELS, is_train=False)

    train_dataset = RainDataset(train_files, transform=train_transform, return_meta=False)
    train_loader = make_loader(train_dataset, shuffle=True)

    val_loader = None
    if val_files:
        val_dataset = RainDataset(val_files, transform=eval_transform, return_meta=False)
        val_loader = make_loader(val_dataset, shuffle=False)

    print("=" * 60)
    print(f"[Data] train_dir = {root_dir}")
    print(f"[Data] 全部样本数 = {len(all_files)}")
    print(f"[Data] 训练样本数 = {len(train_files)}")
    print(f"[Data] 验证样本数 = {len(val_files)}")
    print(f"[Data] 归一化统计样本数 = {len(stat_files)}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_files": train_files,
        "val_files": val_files,
        "mean": mean,
        "std": std,
        "norm_file": norm_path,
    }

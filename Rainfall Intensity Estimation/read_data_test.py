from typing import Dict, Optional

import config
from read_data import RainDataset, build_transform, collect_image_files, load_norm_stats_file, make_loader


def build_test_loader(
    test_dir: Optional[str] = None,
    norm_file: Optional[str] = None,
) -> Dict[str, object]:
    test_dir = config.test_dir if test_dir is None else test_dir
    norm_file = config.NORM_STATS_FILE if norm_file is None else norm_file

    test_files = collect_image_files(test_dir)
    mean, std, norm_path = load_norm_stats_file(norm_file)
    transform = build_transform(mean, std, config.INPUT_CHANNELS, is_train=False)

    test_dataset = RainDataset(test_files, transform=transform, return_meta=True)
    test_loader = make_loader(test_dataset, shuffle=False)

    print("=" * 60)
    print(f"[TestData] test_dir = {test_dir}")
    print(f"[TestData] 测试样本数 = {len(test_files)}")

    return {
        "test_loader": test_loader,
        "test_files": test_files,
        "mean": mean,
        "std": std,
        "norm_file": norm_path,
    }


def get_test_loader():
    return build_test_loader()["test_loader"]

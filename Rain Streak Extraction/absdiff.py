import os
import cv2
from tqdm import tqdm

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def absdiff_and_save_by_name(orig_folder, bg_folder, save_folder, txt_name):
    ensure_dir(save_folder)
    orig_imgs = {f for f in os.listdir(orig_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    bg_imgs = {f for f in os.listdir(bg_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    common_imgs = sorted(orig_imgs & bg_imgs)  # 交集并排序
    used_names = []

    # tqdm可视化进度条
    for name in tqdm(common_imgs, desc='ABSDIFF processing', ncols=80):
        orig_path = os.path.join(orig_folder, name)
        bg_path = os.path.join(bg_folder, name)
        save_path = os.path.join(save_folder, name)
        img1 = cv2.imread(orig_path)
        img2 = cv2.imread(bg_path)
        if img1 is None or img2 is None or img1.shape != img2.shape:
            print(f"跳过: {name}（尺寸不符或文件无法读取）")
            continue
        diff = cv2.absdiff(img1, img2)
        cv2.imwrite(save_path, diff)
        used_names.append(name)

    # 保存名称txt
    parent_dir = os.path.dirname(os.path.abspath(orig_folder))
    txt_path = txt_name if os.path.isabs(txt_name) else os.path.join(parent_dir, txt_name)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for name in used_names:
            f.write(f"{name}\n")
    print(f"\n已完成差分与名称记录。结果保存在: {save_folder}，名称txt: {txt_path}")

if __name__ == "__main__":
    orig_folder = r"E:\Dataset\Forgta\zffevent\original\event7\00009"
    bg_folder = r"E:\Dataset\Forgta\zffevent\derain\event7\00009"
    save_folder = r"E:\Dataset\Forgta\zffevent\rain\event7\00009"
    txt_name = r"E:\Dataset\Forgta\zffevent\rain\event7\00009.txt"
    absdiff_and_save_by_name(orig_folder, bg_folder, save_folder, txt_name)

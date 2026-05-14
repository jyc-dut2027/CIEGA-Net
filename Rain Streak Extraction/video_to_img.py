import cv2
import os
from tqdm import tqdm


def extract_frames(
    video_path,
    output_folder,
    pathlist_name,
    sample_fps=None,          # None=全部抽帧
    start_time=0,
    end_time=None,
    resize=None,              # (W, H)
    resize_ratio=None         # 比例缩放
):
    assert not (resize and resize_ratio), "resize 和 resize_ratio 只能选一个"

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    if end_time is None or end_time > duration:
        end_time = duration

    # ===== 抽帧步长 =====
    if sample_fps is None:
        frame_step = 1
        mode_desc = "全部抽帧"
    else:
        sample_fps = min(sample_fps, video_fps)
        frame_step = max(1, int(round(video_fps / sample_fps)))
        mode_desc = f"{sample_fps} fps 抽帧"

    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"\n🎥 视频: {os.path.basename(video_path)}")
    print(f"抽帧方式: {mode_desc}")
    if resize:
        print(f"缩放尺寸: {resize}")
    elif resize_ratio:
        print(f"缩放比例: {resize_ratio}")
    else:
        print("缩放: 不启用")

    filenames = []
    current_frame = start_frame
    saved_count = 0

    expected = max(1, (end_frame - start_frame) // frame_step)

    with tqdm(total=expected, desc=mode_desc) as pbar:
        while current_frame <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            elif resize_ratio is not None:
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (int(w * resize_ratio), int(h * resize_ratio)),
                    interpolation=cv2.INTER_AREA
                )

            saved_count += 1
            fname = f"{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, fname), frame)
            filenames.append(fname)

            current_frame += frame_step
            pbar.update(1)

    cap.release()

    pathlist_path = os.path.join(output_folder, pathlist_name)
    with open(pathlist_path, "w") as f:
        for name in filenames:
            f.write(name + "\n")

    print(f"\n✅ 保存 {saved_count} 张")
    print(f"📄 pathlist: {pathlist_path}\n")

    return filenames


# ================== 配置区 ==================
video_path = r"D:\Video\ZWY\250923_1412.MOV"
output_dir = r"D:\Image\ZWY\250923_1412\original"

sample_fps = 1          # None / 1 / 5 / 10 ...
resize = None
resize_ratio = 0.5

pathlist_name = "pathlist.txt"

# ================== 主流程 ==================
if __name__ == "__main__":
    extract_frames(
        video_path,
        output_dir,
        pathlist_name,
        sample_fps=sample_fps,
        resize=resize,
        resize_ratio=resize_ratio
    )

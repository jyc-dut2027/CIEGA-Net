import os
import torch

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出目录：模型、归一化统计、训练记录、测试结果和图片都保存在这里
DATA_ROOT = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_ROOT, exist_ok=True)

# 数据目录：公开到 GitHub 后，按需改成自己的本地路径
train_dir = os.path.join(BASE_DIR, "data", "train")
test_dir = os.path.join(BASE_DIR, "data", "test")

# 实验编号与输入设置
EXP_ID = "1"
INPUT_CHANNELS = 1
INPUT_SIZE = (224, 224)
CH_TAG = f"{INPUT_CHANNELS}ch"

# 文件保存路径
NORM_STATS_FILE = os.path.join(DATA_ROOT, f"norm_stats_{CH_TAG}.json")
model_path = os.path.join(DATA_ROOT, f"best_{EXP_ID}_model.pth")
Excel_path = os.path.join(DATA_ROOT, f"test_{EXP_ID}_results.xlsx")
TRAIN_HISTORY_CSV = os.path.join(DATA_ROOT, f"training_history_{EXP_ID}.csv")
LAST_CKPT_PATH = os.path.join(DATA_ROOT, f"last_{EXP_ID}.pth")
INTERRUPT_CKPT_PATH = os.path.join(DATA_ROOT, f"interrupt_{EXP_ID}.pth")
LOSS_CURVE_PATH = os.path.join(DATA_ROOT, f"loss_{EXP_ID}.png")
SCATTER_FIG_PATH = os.path.join(DATA_ROOT, f"prediction_{EXP_ID}.png")

# 图像预处理：训练和测试必须保持一致
USE_RGB_TO_Y = True
USE_SOBEL_EDGE = True
PER_IMAGE_NORMALIZE = True
BINARY_NEAREST_RESIZE = False

# 数据集级归一化：仅当 PER_IMAGE_NORMALIZE=False 时使用
NORM_SCOPE = "train_only"  # 可选："train_only" 或 "all_in_root"
NORM_USE_CACHE = True
FORCE_REBUILD_NORM = False
MAX_STAT_IMAGES = 800  # None 表示使用全部图片统计
BINARY_NORM_POLICY = "compute"  # 可选："compute"、"skip"、"fixed"
BINARY_FIXED_MEAN_STD = ([0.5], [0.5])

# 模型设置
PRETRAINED = True
TORCH_HOME = None  # 例如：r"E:\\RainNet\\ResNet34_model_path"
ENABLE_DROPOUT = True
DROPOUT_P = 0.30

# 设备与 DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = False

# 训练参数
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
HUBER_BETA = 1.0
USE_VALIDATION = True
VAL_RATIO = 0.25
SPLIT_RANDOM_SEED = 3407
RESUME_TRAINING = False
RESUME_CKPT_PATH = None
PROGRESS_COLOR = "red"

# 一致性正则化：真实雨强接近的样本，约束预测值也尽量接近；设为 0.0 表示关闭
CONSISTENCY_LOSS_WEIGHT = 0.0
CONSISTENCY_LOSS_THRESHOLD = 2.0

# 训练 loss 曲线
PLOT_LOSS_CURVE = True
LOSS_CURVE_FIGSIZE = (6, 4)
LOSS_CURVE_DPI = 150

# 测试散点图
PLOT_TEST_SCATTER = True
SCATTER_FIG_FIGSIZE = (6, 6)
SCATTER_FIG_DPI = 150
SCATTER_TEXT_POS = (0.68, 0.20)
SCATTER_TEXT_FONTSIZE = 10

import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
import Net
from read_data_test import build_test_loader
from utils import compute_kge, compute_mae, compute_mape, compute_nse, compute_r2, compute_rmse


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_torch_load(model_path: str, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def build_model(device):
    dropout_p = config.DROPOUT_P if config.ENABLE_DROPOUT else 0.0
    return Net.RainNet(
        in_channels=config.INPUT_CHANNELS,
        dropout_p=dropout_p,
        pretrained=config.PRETRAINED,
        torch_home=config.TORCH_HOME,
    ).to(device)


def load_trained_model(model_path: str, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型参数文件: {model_path}")

    model = build_model(device)
    state = _safe_torch_load(model_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def compute_metrics(predictions, targets, device):
    pred = torch.tensor(predictions, device=device, dtype=torch.float32)
    tgt = torch.tensor(targets, device=device, dtype=torch.float32)
    return {
        "MAE(mm/h)": compute_mae(pred, tgt).item(),
        "MAPE(%)": compute_mape(pred, tgt).item(),
        "RMSE(mm/h)": compute_rmse(pred, tgt).item(),
        "r²": compute_r2(pred, tgt).item(),
        "NSE": compute_nse(pred, tgt).item(),
        "KGE": compute_kge(pred, tgt).item(),
    }


def plot_prediction_scatter(
    results_df: pd.DataFrame,
    metrics: dict,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
):
    if not config.PLOT_TEST_SCATTER:
        return None

    out_path = config.SCATTER_FIG_PATH if out_path is None else out_path
    _ensure_parent(out_path)

    observed = results_df["观测值(mm/h)"].to_numpy(dtype=float)
    predicted = results_df["预测值(mm/h)"].to_numpy(dtype=float)

    plt.figure(figsize=config.SCATTER_FIG_FIGSIZE)
    plt.scatter(observed, predicted, alpha=0.75)

    min_v = float(min(np.min(observed), np.min(predicted)))
    max_v = float(max(np.max(observed), np.max(predicted)))
    if np.isclose(min_v, max_v):
        min_v -= 1.0
        max_v += 1.0

    plt.plot([min_v, max_v], [min_v, max_v], linestyle="-")
    plt.xlabel("Observation (mm/h)")
    plt.ylabel("Prediction (mm/h)")
    plt.title(title or f"Prediction vs Observation - {config.EXP_ID}")
    plt.grid(True, alpha=0.3)

    text = (
        f"MAE={metrics['MAE(mm/h)']:.2f} mm/h\n"
        f"RMSE={metrics['RMSE(mm/h)']:.2f} mm/h\n"
        f"MAPE={metrics['MAPE(%)']:.2f}%\n"
        f"r²={metrics['r²']:.3f}\n"
        f"NSE={metrics['NSE']:.3f}\n"
        f"KGE={metrics['KGE']:.3f}"
    )
    plt.gcf().text(
        config.SCATTER_TEXT_POS[0],
        config.SCATTER_TEXT_POS[1],
        text,
        fontsize=config.SCATTER_TEXT_FONTSIZE,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=config.SCATTER_FIG_DPI)
    plt.close()
    print(f"[Plot] 散点图已保存: {out_path}")
    return out_path


@torch.no_grad()
def predict_loader(model, test_loader, device) -> pd.DataFrame:
    model.eval()
    rows = []

    for features, seq_nums, event_ids, targets, fnames in test_loader:
        features = features.to(device)
        targets = targets.to(device).float().view(-1)
        predictions = model(features).float().view(-1)
        abs_errors = torch.abs(predictions - targets)

        for seq, event_id, target, prediction, abs_error, fname in zip(
            seq_nums,
            event_ids,
            targets.cpu().numpy().tolist(),
            predictions.cpu().numpy().tolist(),
            abs_errors.cpu().numpy().tolist(),
            fnames,
        ):
            rows.append({
                "序号": int(seq),
                "观测值(mm/h)": float(target),
                "预测值(mm/h)": float(prediction),
                "文件名": fname,
                "事件编号": int(event_id),
                "绝对误差(mm/h)": float(abs_error),
            })

    if not rows:
        raise ValueError("测试结果为空。")

    df = pd.DataFrame(rows).sort_values(by=["事件编号", "序号"]).reset_index(drop=True)
    first_cols = ["序号", "观测值(mm/h)", "预测值(mm/h)"]
    other_cols = [col for col in df.columns if col not in first_cols]
    return df[first_cols + other_cols]


def evaluate_test_set(
    test_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    norm_file: Optional[str] = None,
    excel_path: Optional[str] = None,
    scatter_path: Optional[str] = None,
    device=None,
    title: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    device = config.DEVICE if device is None else device
    test_dir = config.test_dir if test_dir is None else test_dir
    model_path = config.model_path if model_path is None else model_path
    norm_file = config.NORM_STATS_FILE if norm_file is None else norm_file
    excel_path = config.Excel_path if excel_path is None else excel_path
    scatter_path = config.SCATTER_FIG_PATH if scatter_path is None else scatter_path

    _ensure_parent(excel_path)
    _ensure_parent(scatter_path)

    print("=" * 60)
    print(f"[Test] device = {device}")
    print(f"[Test] test_dir = {test_dir}")
    print(f"[Test] model_path = {model_path}")
    print(f"[Test] norm_file = {norm_file}")

    data_pack = build_test_loader(test_dir=test_dir, norm_file=norm_file)
    model = load_trained_model(model_path, device=device)
    results_df = predict_loader(model, data_pack["test_loader"], device=device)

    metrics = compute_metrics(
        predictions=results_df["预测值(mm/h)"].tolist(),
        targets=results_df["观测值(mm/h)"].tolist(),
        device=device,
    )

    summary_df = pd.DataFrame([{
        "样本数": len(results_df),
        **metrics,
        "模型文件": model_path,
        "测试目录": test_dir,
        "归一化文件": norm_file,
    }])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="sample_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"[Excel] 测试结果已保存: {excel_path}")
    print(summary_df.drop(columns=["模型文件", "测试目录", "归一化文件"]).to_string(index=False))

    plot_prediction_scatter(
        results_df=results_df,
        metrics=metrics,
        out_path=scatter_path,
        title=title,
    )

    return results_df, summary_df


def main():
    evaluate_test_set()


if __name__ == "__main__":
    main()

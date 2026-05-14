import datetime
import os
import time
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
import Net
from read_data import build_train_val_loaders
from utils import compute_kge, compute_mae, compute_mape, compute_nse, compute_r2, huber_loss


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_model(device):
    dropout_p = config.DROPOUT_P if config.ENABLE_DROPOUT else 0.0
    return Net.RainNet(
        in_channels=config.INPUT_CHANNELS,
        dropout_p=dropout_p,
        pretrained=config.PRETRAINED,
        torch_home=config.TORCH_HOME,
    ).to(device)


def _loss_and_metrics(predictions, targets):
    predictions = predictions.view(-1).float()
    targets = targets.view(-1).float()
    valid = ~torch.isnan(predictions) & ~torch.isnan(targets)
    predictions = predictions[valid]
    targets = targets[valid]

    if len(predictions) == 0:
        zero = torch.tensor(0.0, device=targets.device)
        return zero, (0.0, 0.0, 0.0, 0.0, 0.0)

    loss = huber_loss(predictions, targets, delta=config.HUBER_BETA)
    metrics = (
        compute_mae(predictions, targets).item(),
        compute_mape(predictions, targets).item(),
        compute_r2(predictions, targets).item(),
        compute_nse(predictions, targets).item(),
        compute_kge(predictions, targets).item(),
    )
    return loss, metrics


def consistency_loss(predictions, targets, threshold: float):
    predictions = predictions.view(-1).float()
    targets = targets.view(-1).float()
    valid = ~torch.isnan(predictions) & ~torch.isnan(targets)
    predictions = predictions[valid]
    targets = targets[valid]

    if len(predictions) < 2:
        return torch.tensor(0.0, device=predictions.device)

    target_diff = torch.abs(targets[:, None] - targets[None, :])
    pred_diff = predictions[:, None] - predictions[None, :]
    mask = torch.triu(target_diff < threshold, diagonal=1)

    if not torch.any(mask):
        return torch.tensor(0.0, device=predictions.device)
    return torch.mean(pred_diff[mask] ** 2)


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    model.eval()
    if loader is None:
        return 0, 0, 0, 0, 0, 0

    rows = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device).view(-1).float()
        predictions = model(features).float().view(-1)
        loss, metrics = _loss_and_metrics(predictions, targets)
        rows.append((loss.item(), *metrics))

    if not rows:
        return 0, 0, 0, 0, 0, 0
    return tuple(np.mean(rows, axis=0).tolist())


def save_loss_curve(history_df: pd.DataFrame, out_path: Optional[str] = None):
    if not config.PLOT_LOSS_CURVE:
        return None

    out_path = config.LOSS_CURVE_PATH if out_path is None else out_path
    _ensure_parent(out_path)

    plt.figure(figsize=config.LOSS_CURVE_FIGSIZE)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")

    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {config.EXP_ID}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=config.LOSS_CURVE_DPI)
    plt.close()

    print(f"[Plot] loss 曲线已保存: {out_path}")
    return out_path


def _save_checkpoint(path, epoch, model, optimizer, scheduler, best_metric, history_rows):
    _ensure_parent(path)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history_rows": history_rows,
        "exp_id": config.EXP_ID,
    }, path)


def _load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    history_rows = list(checkpoint.get("history_rows", []))
    return start_epoch, best_metric, history_rows


def train_model(
    train_loader,
    val_loader=None,
    device=None,
    model_path: Optional[str] = None,
    history_csv_path: Optional[str] = None,
    loss_curve_path: Optional[str] = None,
    last_ckpt_path: Optional[str] = None,
) -> Tuple[str, pd.DataFrame]:
    device = config.DEVICE if device is None else device
    model_path = config.model_path if model_path is None else model_path
    history_csv_path = config.TRAIN_HISTORY_CSV if history_csv_path is None else history_csv_path
    loss_curve_path = config.LOSS_CURVE_PATH if loss_curve_path is None else loss_curve_path
    last_ckpt_path = config.LAST_CKPT_PATH if last_ckpt_path is None else last_ckpt_path

    _ensure_parent(model_path)
    _ensure_parent(history_csv_path)

    model = build_model(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 1
    best_metric = float("inf")
    history_rows = []

    if config.RESUME_TRAINING:
        resume_path = config.RESUME_CKPT_PATH or last_ckpt_path
        if resume_path and os.path.isfile(resume_path):
            start_epoch, best_metric, history_rows = _load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            print(f"[Resume] 从断点继续训练: {resume_path}, start_epoch={start_epoch}")
        else:
            print(f"[Resume] 未找到断点文件，重新开始训练: {resume_path}")

    tz = datetime.timezone(datetime.timedelta(hours=8))
    print(f"[Start] {datetime.datetime.now(tz)}")
    print(f"[Device] {device}")
    start_time = time.time()
    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
            model.train()
            epoch_losses = []

            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch}/{config.NUM_EPOCHS}",
                colour=config.PROGRESS_COLOR,
            )

            for features, targets in progress:
                features = features.to(device)
                targets = targets.to(device).view(-1).float()

                optimizer.zero_grad()
                predictions = model(features).float().view(-1)
                base_loss, _ = _loss_and_metrics(predictions, targets)

                if config.CONSISTENCY_LOSS_WEIGHT > 0:
                    reg_loss = consistency_loss(
                        predictions,
                        targets,
                        threshold=config.CONSISTENCY_LOSS_THRESHOLD,
                    )
                    loss = base_loss + config.CONSISTENCY_LOSS_WEIGHT * reg_loss
                else:
                    loss = base_loss

                if not torch.isnan(loss) and loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                epoch_losses.append(float(loss.item()))

            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            row = {"epoch": epoch, "train_loss": avg_train_loss}

            if config.USE_VALIDATION and val_loader is not None:
                val_loss, v_mae, v_mape, v_r2, v_nse, v_kge = evaluate_on_loader(model, val_loader, device)
                row.update({
                    "val_loss": val_loss,
                    "val_mae": v_mae,
                    "val_mape": v_mape,
                    "val_r2": v_r2,
                    "val_nse": v_nse,
                    "val_kge": v_kge,
                })
                metric = val_loss
                print(
                    f"[Epoch {epoch}] "
                    f"TrainLoss={avg_train_loss:.4f} | "
                    f"ValLoss={val_loss:.4f} | "
                    f"Val(MAE/MAPE/r²/NSE/KGE)=({v_mae:.2f}/{v_mape:.2f}/{v_r2:.3f}/{v_nse:.3f}/{v_kge:.3f})"
                )
            else:
                metric = avg_train_loss
                row["val_loss"] = np.nan
                print(f"[Epoch {epoch}] TrainLoss={avg_train_loss:.4f}")

            history_rows.append(row)

            if metric < best_metric:
                best_metric = metric
                torch.save(model.state_dict(), model_path)
                print(f"  -> 保存最优模型 @ epoch {epoch}: {model_path}")

            scheduler.step(metric)
            _save_checkpoint(last_ckpt_path, epoch, model, optimizer, scheduler, best_metric, history_rows)

    except KeyboardInterrupt:
        _save_checkpoint(config.INTERRUPT_CKPT_PATH, epoch, model, optimizer, scheduler, best_metric, history_rows)
        print(f"[Interrupt] 已保存中断断点: {config.INTERRUPT_CKPT_PATH}")
        raise

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    save_loss_curve(history_df, out_path=loss_curve_path)

    elapsed = time.time() - start_time
    print(f"[Done] best_metric={best_metric:.4f}, elapsed={elapsed / 60:.1f} min")
    print(f"[Done] 模型已保存: {model_path}")
    print(f"[Done] 训练记录已保存: {history_csv_path}")

    return model_path, history_df


def main():
    data_pack = build_train_val_loaders(config.train_dir)
    train_model(
        train_loader=data_pack["train_loader"],
        val_loader=data_pack["val_loader"],
        device=config.DEVICE,
    )


if __name__ == "__main__":
    main()

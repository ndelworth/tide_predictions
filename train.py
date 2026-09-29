import numpy as np
import pandas as pd
import torch
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader

from config import MODEL_PATH, TRAIN_END
from features import WINDOW_OFFSETS, ForecastDataset
from model import SurgeTransformer, device, persistence

_last_day = TRAIN_END.ceil("D")
VAL_BLOCKS = [
    ("2024-04-01", "2024-04-22"),
    ("2025-02-01", "2025-02-22"),
    ("2025-04-01", "2025-04-22"),
    ("2026-02-01", "2026-02-22"),
    (_last_day - pd.DateOffset(months=2), _last_day),
]

FIT_EVERY_NTH_HOUR = 3
BATCH_SIZE         = 64
LR                 = 1.5e-4
WEIGHT_DECAY       = 5e-4
EPOCHS             = 100
PATIENCE           = 20
GRAD_CLIP          = 1.0
WEATHER_NOISE_STD  = 0.05
SEED               = 0


def in_validation(times):
    return np.any([(times >= start) & (times < end) for start, end in VAL_BLOCKS], axis=0)


def split(data, in_val):
    issued = np.flatnonzero(data.usable)
    window_touches_val = in_val[issued[:, None] + WINDOW_OFFSETS].any(axis=1)
    fit = issued[~window_touches_val & (issued % FIT_EVERY_NTH_HOUR == 0)]
    val = issued[in_val[issued]]
    return fit, val


def observed_l1(pred, target, reduction="mean"):
    observed = ~target.isnan()
    return l1_loss(pred[observed], target[observed], reduction=reduction)


def train_one_epoch(model, loader, optimizer):
    model.train()
    losses = []
    for batch in loader:
        weather, surge, time, target = (x.to(device) for x in batch)
        weather = weather + torch.randn_like(weather) * WEATHER_NOISE_STD * model.weather_scale.std
        loss = observed_l1(model(weather, surge, time), target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def validation_mae(predict, loader):
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            weather, surge, time, target = (x.to(device) for x in batch)
            total += observed_l1(predict(weather, surge, time), target, reduction="sum").item()
            n += (~target.isnan()).sum().item()
    return total / n


def main():
    torch.manual_seed(SEED)
    data = ForecastDataset(through=TRAIN_END)
    in_val = in_validation(data.times)
    fit_issued, val_issued = split(data, in_val)
    print(f"{len(data.times):,} training-period hours; {len(fit_issued):,} fit windows, "
          f"{len(val_issued):,} validation windows in {len(VAL_BLOCKS)} blocks; device {device}")
    fit = DataLoader(data.windows(fit_issued), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val = DataLoader(data.windows(val_issued), batch_size=256)

    model = SurgeTransformer()
    model.fit_standardization(data.df[~in_val])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    baseline = validation_mae(lambda weather, surge, time: persistence(surge), val)
    print(f"Validation MAE of plain persistence: {baseline * 100:.2f} cm\n")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    best, best_epoch = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch(model, fit, optimizer)
        scheduler.step()
        model.eval()
        val_mae = validation_mae(model, val)
        improved = val_mae < best
        if improved:
            best, best_epoch = val_mae, epoch
            torch.save(model.state_dict(), MODEL_PATH)
        print(f"epoch {epoch:3d}  train {train_mae * 100:6.2f} cm  val {val_mae * 100:6.2f} cm"
              + ("  <- best, saved" if improved else ""))
        if epoch - best_epoch >= PATIENCE:
            break

    print(f"\nBest validation MAE {best * 100:.2f} cm (epoch {best_epoch}; persistence "
          f"{baseline * 100:.2f} cm) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

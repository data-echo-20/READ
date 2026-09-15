# -*- coding: utf-8 -*-
"""训练入口（论文算法 2 + 第 5.3 节超参数）。

运行：python train.py --data /path/to/data --out best_model.pt
数据目录格式见 data.py 开头的数据组织说明。
"""
import argparse
import math
import random

import torch
from torch.utils.data import DataLoader

import config
from model import CIPDatingModel
from losses import total_loss
from data import CIPDatingDataset, build_tokenizer, collate_fn


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grid():
    return torch.linspace(config.T_MIN, config.T_MAX, config.L)


def cosine_annealing_with_warmup(optimizer, epoch: int, warmup: int, total: int):
    """5 轮线性预热 + 余弦退火（第 5.3 节）。"""
    if epoch < warmup:
        lr = config.LR * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr = config.LR * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def evaluate(model, loader, grid, device):
    model.eval()
    mae, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            out = model(image, input_ids, attn)
            mae += (out['y_hat'] - batch['year'].to(device)).abs().sum().item()
            correct += (out['p_class'].argmax(-1) == batch['dynasty'].to(device)).sum().item()
            total += image.shape[0]
    return mae / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='数据根目录，含 inscriptions.json 与 images/')
    ap.add_argument('--out', default='best_model.pt', help='最优权重保存路径')
    ap.add_argument('--epochs', type=int, default=config.EPOCHS)
    ap.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--text-model', default=config.TEXT_MODEL,
                    help='SikuBERT 的 HuggingFace id 或本地权重目录')
    ap.add_argument('--freeze-text', action='store_true',
                    help='冻结题跋编码器，只训练下游头')
    args = ap.parse_args()

    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid = build_grid().to(device)

    model = CIPDatingModel(grid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                  betas=config.ADAM_BETAS,
                                  weight_decay=config.WEIGHT_DECAY)

    tokenizer = build_tokenizer(args.text_model)
    # 训练集开启题跋增广（丢题跋 / 跨朝代伪题，论文 5.3 节），验证集不增广
    train_ds = CIPDatingDataset(args.data, split='train', tokenizer=tokenizer,
                                augment=True)
    val_ds = CIPDatingDataset(args.data, split='val', tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.workers)
    print(f'训练集 {len(train_ds)} 件 | 验证集 {len(val_ds)} 件 | 设备 {device}')

    best_mae, patience = float('inf'), 0
    for epoch in range(args.epochs):
        lr = cosine_annealing_with_warmup(optimizer, epoch, config.WARMUP_EPOCHS,
                                          args.epochs)
        model.train()
        for batch in train_loader:
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            year = batch['year'].to(device)
            dynasty = batch['dynasty'].to(device)
            is_consistent = batch['is_consistent'].to(device)

            optimizer.zero_grad()
            out = model(image, input_ids, attn)
            loss, stats = total_loss(out, year, dynasty, is_consistent, grid)
            loss.backward()
            optimizer.step()

        val_mae, val_acc = evaluate(model, val_loader, grid, device)
        print(f"epoch {epoch+1:3d} | lr {lr:.2e} | val_mae {val_mae:6.2f} | "
              f"val_acc {val_acc*100:5.2f}% | reg {stats['l_reg']:.3f} "
              f"cls {stats['l_cls']:.3f} cons {stats['l_cons']:.3f}")

        if val_mae < best_mae:
            best_mae, patience = val_mae, 0
            torch.save(model.state_dict(), args.out)
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"early stop at epoch {epoch+1} (best val_mae {best_mae:.2f})")
                break


if __name__ == '__main__':
    main()

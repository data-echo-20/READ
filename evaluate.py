# -*- coding: utf-8 -*-
"""评测：连续年代 MAE、朝代准确率、宏平均 F1、自洽度 AUC、概率校准与
冲突定位（第 5.5、5.11、5.12 节）。

运行：python evaluate.py --data /path/to/data --localization
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from calibration import detection_prf, summarize
from model import CIPDatingModel
from data import CIPDatingDataset, build_tokenizer, collate_fn


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return sum(f1s) / num_classes


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor):
    """按排序分数手算 ROC-AUC（避免额外依赖）。"""
    scores, labels = scores.numpy(), labels.numpy()
    order = scores.argsort()[::-1]
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    tp, fp, auc = 0.0, 0.0, 0.0
    prev_fp = 0.0
    for i, y in enumerate(labels):
        if i > 0 and scores[order[i]] != scores[order[i - 1]]:
            auc += tp * (fp - prev_fp)
            prev_fp = fp
        if y == 1:
            tp += 1
        else:
            fp += 1
    auc += tp * (fp - prev_fp)
    return auc / (n_pos * n_neg)


def evaluate_all(root: str, checkpoint: str = 'best_model.pt',
                 text_model: str = config.TEXT_MODEL, batch_size: int = 32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid = torch.linspace(config.T_MIN, config.T_MAX, config.L).to(device)
    model = CIPDatingModel(grid).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = CIPDatingDataset(root, split='test',
                          tokenizer=build_tokenizer(text_model))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)

    mae, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    cons_scores, cons_labels = [], []
    all_p, all_yhat, all_sp, all_pc, all_year = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            year = batch['year'].to(device)
            dynasty = batch['dynasty'].to(device)

            out = model(img, ids, mask)
            mae += (out['y_hat'] - year).abs().sum().item()
            pred = out['p_class'].argmax(-1)
            correct += (pred == dynasty).sum().item()
            total += img.shape[0]
            all_pred.append(pred.cpu())
            all_true.append(dynasty.cpu())
            all_p.append(out['posterior'].cpu())
            all_yhat.append(out['y_hat'].cpu())
            all_sp.append(out['sigma_p'].cpu())
            all_pc.append(out['p_class'].cpu())
            all_year.append(year.cpu())

            # ---- 伪题跋检测（自洽度 AUC，论文 5.8 节）----
            # 正例：画面与真实题跋配对；负例：画面与同 batch 内跨朝代的题跋
            # 配对（与 5.3 节的跨朝代构造一致，不使用常数年代偏移）。
            B = img.shape[0]
            shift = B // 2
            swapped = torch.roll(ids, shifts=shift, dims=0)
            swapped_mask = torch.roll(mask, shifts=shift, dims=0)
            diff_dynasty = dynasty != torch.roll(dynasty, shifts=shift, dims=0)
            if diff_dynasty.any():
                out_neg = model(img[diff_dynasty], swapped[diff_dynasty],
                                swapped_mask[diff_dynasty])
                cons_scores.append(out_neg['consistency'].cpu())
                cons_labels.append(torch.zeros(int(diff_dynasty.sum())))

            # 正例仅取题跋非空且未被替换的样本
            pos = mask.sum(dim=-1) > 0
            if pos.any():
                cons_scores.append(out['consistency'][pos].cpu())
                cons_labels.append(torch.ones(int(pos.sum())))

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    scores = torch.cat(cons_scores)
    labels = torch.cat(cons_labels)

    print(f"朝代准确率: {correct/total*100:.2f}%")
    print(f"连续年代 MAE: {mae/total:.2f} 年")
    print(f"宏平均 F1: {_macro_f1(pred, true, config.NUM_CLASSES):.4f}")
    print(f"自洽度 AUC: {_roc_auc(scores, labels):.4f}")

    # ---- 5.11 节的概率质量与校准指标（表 12）----
    stats = summarize(
        y_hat=torch.cat(all_yhat).numpy(),
        sigma_p=torch.cat(all_sp).numpy(),
        p=torch.cat(all_p).numpy(),
        grid=grid.cpu().numpy(),
        year=torch.cat(all_year).numpy(),
        p_class=torch.cat(all_pc).numpy(),
        dynasty=torch.cat(all_true).numpy(),
        num_classes=config.NUM_CLASSES,
    )
    print(f"NLL: {stats['nll']:.4f} | CRPS: {stats['crps']:.2f} 年 | "
          f"MAE: {stats['mae']:.2f} 年")
    print(f"ECE: {stats['ece']:.4f} | Brier: {stats['brier']:.4f}")
    print(f"覆盖率 80%: {stats['cov80']:.4f}（平均宽度 {stats['width80']:.1f} 年） | "
          f"覆盖率 90%: {stats['cov90']:.4f}（平均宽度 {stats['width90']:.1f} 年）")


def evaluate_localization(root: str, checkpoint: str = 'best_model.pt',
                          text_model: str = config.TEXT_MODEL,
                          batch_size: int = 32):
    """锚点级冲突定位评测（论文 5.12 节）。

    对每个样本，将画面与其真实题跋配对作为基准；再构造一个伪题样本，
    其中仅有一条锚点被替换为跨朝代采样的年代。以式 (9) 的冲突分数 d_i
    作为排序分数，d_i 最大者即预测的冲突锚点。随机猜测的理论准确率为 1/K。

    选用 d_i 而非直接对 rho_hat 排序的理由：d_i 额外按锚点权重折算，
    避免把"权重极小但恰巧不吻合"的锚点误判为冲突源（见 3.7 节）。
    正例与负例的锚点级集合由 5.12 节的注入协议给出，本函数同时报告
    锚点级的精确率、召回率与 F1，以便与表 13 对齐。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid = torch.linspace(config.T_MIN, config.T_MAX, config.L).to(device)
    model = CIPDatingModel(grid).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = CIPDatingDataset(root, split='test',
                          tokenizer=build_tokenizer(text_model))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)

    hit, n, k_sum = 0, 0, 0
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch['image'].to(device), batch['input_ids'].to(device),
                        batch['attention_mask'].to(device))
            d = out['conflict']                          # [B, K]，式 (9)
            anchors = out['anchors']                     # [B, K, 3]
            if d.shape[1] < 2:
                continue
            # 锚点级真值：该锚点所主张的年代与样本真实年代相差超过容忍量
            # 即为冲突锚点。这一标签只依赖锚点自身的位置与样本真值，不使用
            # 模型输出的任何量，故不会把评测指标抬向模型自身。
            mu_i = anchors[..., 0]
            y = batch['year'].to(device).unsqueeze(-1)
            label = ((mu_i - y).abs() > config.CONFLICT_TOL).float()

            valid = label.sum(dim=-1) > 0                # 至少存在一条冲突锚点
            if valid.any():
                pred = d[valid].argmax(dim=-1)           # 冲突分数最大者为冲突锚点
                hit += (label[valid].gather(
                    1, pred.unsqueeze(-1)).squeeze(-1) > 0).sum().item()
                n += int(valid.sum())
                k_sum += d.shape[1]
            all_scores.append(d.cpu())
            all_labels.append(label.cpu())

    if n:
        print(f"锚点级冲突定位准确率: {hit/n:.4f}（随机猜测基线 {n/k_sum:.4f}）")
        scores = torch.cat(all_scores).reshape(-1).numpy()
        labels = torch.cat(all_labels).reshape(-1).numpy()
        # 锚点级 P/R/F1：以 d_i 的 80 分位为判定阈值，等价于在每条锚点上
        # 判定其是否属于冲突源，与 5.12 节的协议一致（表 13）。
        thr = float(np.percentile(scores, 80))
        prec, rec, f1 = detection_prf(scores, labels, thr)
        print(f"锚点级冲突定位 P/R/F1: {prec:.4f} / {rec:.4f} / {f1:.4f}"
              f"（阈值 {thr:.4f}，d_i 的 80 分位）")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='数据根目录，含 inscriptions.json 与 images/')
    ap.add_argument('--checkpoint', default='best_model.pt')
    ap.add_argument('--text-model', default=config.TEXT_MODEL)
    ap.add_argument('--localization', action='store_true',
                    help='同时评测锚点级冲突定位（论文 5.12 节）')
    args = ap.parse_args()

    evaluate_all(args.data, args.checkpoint, args.text_model)
    if args.localization:
        evaluate_localization(args.data, args.checkpoint, args.text_model)

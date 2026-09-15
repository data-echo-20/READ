# -*- coding: utf-8 -*-
"""CIP-Dating 数据集加载与数据增广（论文第 5.1、5.3 节）。

数据组织
--------
数据根目录下放置图像目录与一份 JSON 清单：

    <root>/
      images/
        0001.jpg
        ...
      inscriptions.json

``inscriptions.json`` 为对象数组，每项字段如下：

    id           作品编号，字符串，用于复现划分与报错定位
    image        相对 <root> 的图像路径
    year         作品年代，单位「年」，须落在 [t_min, t_max] 内
    inscription  著录题跋全文；无题跋的作品置为空串
    split        'train' / 'val' / 'test' 之一，缺省视为 'train'

清单的构建与标注协议见论文 5.1 节，本模块只负责读取。
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

import config


def build_tokenizer(name_or_path: str = config.TEXT_MODEL):
    """加载题跋分词器（论文 3.5 节）。

    默认加载 SikuBERT，即在《四库全书》语料上继续预训练的古汉语语言模型
    （王东波, 刘畅, 朱子赫, 等. SikuBERT 与 SikuRoBERTa：面向数字人文的
    《四库全书》预训练模型构建及应用研究[J]. 图书馆论坛, 2022, 42(6): 31-43）。
    也可传入本地权重目录以离线使用。
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name_or_path)


def build_transform(train: bool = False):
    """图像预处理（论文表 4 的输入分辨率一行）。

    训练时随机裁剪，评测时缩放后中心裁剪，输出均为 224×224 并做 ImageNet
    归一化。不做水平翻转：书画的构图与题跋位置均携带年代线索，翻转会引入
    与标注无关的形变。5.3 节的题跋增广在 __getitem__ 中另行施加。
    """
    from torchvision import transforms
    if train:
        crop = transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0),
                                            ratio=(3 / 4, 4 / 3))
    else:
        crop = transforms.Compose([
            transforms.Resize(int(config.IMG_SIZE * 256 / 224)),
            transforms.CenterCrop(config.IMG_SIZE),
        ])
    return transforms.Compose([
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _dynasty_of_year(year: float):
    """由连续年代定位朝代区间，即式 (11) 的积分区间。"""
    for idx, (_, a, b) in enumerate(config.DYNASTY_INTERVALS):
        if a <= year < b:
            return idx
    return config.NUM_CLASSES - 1


class CIPDatingDataset(Dataset):
    """CIP-Dating 书画-题跋成对数据集（论文 5.1 节）。

    每个样本由一幅书画图像、一段著录题跋文本与一个连续年代标注构成。
    朝代标签由年代经 config.DYNASTY_INTERVALS 导出而不单独标注，以免
    两套标签在朝代边界上互相矛盾。

    augment=True 时按论文 5.3 节对题跋施以增广：以 config.DROP_TEXT_PROB
    的概率丢弃题跋，对应零条锚点与恒等似然；以 config.PSEUDO_PROB 的概率
    替换为另一朝代作品的真实题跋，并标记为冲突样本。两种增广互斥。
    """

    def __init__(self, root, split: str = 'train', tokenizer=None,
                 max_len: int = config.MAX_TEXT_LEN, transform=None,
                 augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.transform = (transform if transform is not None
                          else build_transform(train=split == 'train'))
        self.augment = bool(augment) and split == 'train'

        manifest = self.root / 'inscriptions.json'
        if not manifest.exists():
            raise FileNotFoundError(
                f'未找到数据清单 {manifest}；清单格式见本模块开头的数据组织说明。')
        with manifest.open(encoding='utf-8') as f:
            records = json.load(f)

        self.items = [r for r in records if r.get('split', 'train') == split]
        if not self.items:
            raise ValueError(f'清单 {manifest} 中没有 split={split!r} 的作品')

        # 伪题从训练集其它朝代的真实题跋中抽取，故只对训练集分桶，
        # 以免验证集与测试集的题跋经增广泄漏进训练。
        self._pseudo_pool = {}
        for r in records:
            if r.get('split', 'train') != 'train':
                continue
            text = (r.get('inscription') or '').strip()
            if text:
                self._pseudo_pool.setdefault(
                    _dynasty_of_year(float(r['year'])), []).append(text)

    def __len__(self):
        return len(self.items)

    def _encode(self, text: str):
        """题跋文本映射为 (input_ids, attention_mask)，长度固定为 max_len。

        空文本对应无题跋作品，返回全零掩码；融合模块据掩码之和判定有效
        锚点数为零，从而启用恒等似然 r_l ≡ 1（论文 3.7 节）。
        """
        if not text:
            zeros = torch.zeros(self.max_len, dtype=torch.long)
            return zeros, zeros.clone()
        if self.tokenizer is None:
            raise RuntimeError(
                '数据集未配置分词器。请先调用 build_tokenizer()，并把结果经 '
                'CIPDatingDataset(tokenizer=...) 传入。')
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                             padding='max_length', return_tensors='pt')
        return enc['input_ids'][0], enc['attention_mask'][0]

    def _pseudo_inscription(self, true_dynasty: int):
        """为伪题另取一段真实题跋（论文 5.3 节）。

        取自与真实朝代不同的作品，使伪题的年代边缘分布与真实数据一致。
        这里刻意不使用「对真实年代加常数偏移」的构造方式：常数偏移会产生
        年代分布明显偏离真实数据的伪样本，模型便可借分布偏移而非时间不
        一致来完成判别，从而高估自洽度的实际鉴别力。
        """
        pool = [d for d in self._pseudo_pool if d != true_dynasty]
        if not pool:
            return None
        return random.choice(self._pseudo_pool[random.choice(pool)])

    def __getitem__(self, idx: int):
        item = self.items[idx]
        year = float(item['year'])
        dynasty = _dynasty_of_year(year)

        text = (item.get('inscription') or '').strip()
        is_consistent = 1.0
        if self.augment and text:
            r = random.random()
            if r < config.DROP_TEXT_PROB:
                text = ''                                  # 零条锚点，恒等似然
            elif r < config.DROP_TEXT_PROB + config.PSEUDO_PROB:
                pseudo = self._pseudo_inscription(dynasty)
                if pseudo is not None:
                    text, is_consistent = pseudo, 0.0

        image = Image.open(self.root / item['image']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        input_ids, attention_mask = self._encode(text)

        return {'image': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'year': torch.tensor(year, dtype=torch.float32),
                'dynasty': torch.tensor(dynasty, dtype=torch.long),
                'is_consistent': torch.tensor(is_consistent, dtype=torch.float32),
                'id': item['id']}


def collate_fn(batch):
    """把样本列表堆叠为 batch。题跋增广已在 __getitem__ 内完成。"""
    return {'image': torch.stack([b['image'] for b in batch]),
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'year': torch.stack([b['year'] for b in batch]),
            'dynasty': torch.stack([b['dynasty'] for b in batch]),
            'is_consistent': torch.stack([b['is_consistent'] for b in batch]),
            'id': [b['id'] for b in batch]}

"""
基础时序滑窗 Dataset — 所有数据集的父类
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    基础时序数据集，负责：
    1. 加载CSV
    2. 数据标准化
    3. 滑窗构建 (seq_len + label_len + pred_len)
    4. 时间特征提取 (月/日/周/时)
    """

    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', target='OT', scale=True, freq='h',
                 seq_len=96, label_len=48, pred_len=96,
                 text_embeds=None, img_tensors=None):
        """
        Args:
            root_path: 数据集根目录
            data_path: CSV文件名
            flag: 'train' / 'val' / 'test'
            size: (seq_len, label_len, pred_len) 或 None
            features: 'M' / 'S' / 'MS'
            target: 目标变量名 (features='S'或'MS'时使用)
            scale: 是否标准化
            freq: 时间频率
            text_embeds: 预计算的文本嵌入 [N, text_dim] 或 None
            img_tensors: 预计算的递归图 [N, 1, img_size, img_size] 或 None
        """
        if size is None:
            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.flag = flag

        # 数据划分比例 (按时间顺序)
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.border_ratios = [0.0, 0.7, 0.85, 1.0]

        self.data_path = os.path.join(root_path, data_path)
        self.__read_data__()

        self.text_embeds = text_embeds
        self.img_tensors = img_tensors

    def __read_data__(self):
        """读取CSV并做标准化"""
        df = pd.read_csv(self.data_path)

        # 提取目标列索引
        cols = list(df.columns)
        cols.remove('date')
        if self.features == 'S':       # 单变量
            df = df[['date', self.target]]
            cols = [self.target]
        elif self.features == 'MS':    # 多变量预测单变量
            target_col = cols.index(self.target)

        data = df[cols].values.astype(np.float32)

        # 按比例划分
        n = len(data)
        border1 = int(n * self.border_ratios[self.set_type])
        border2 = int(n * self.border_ratios[self.set_type + 1])

        # 用训练集的统计量做标准化
        train_end = int(n * self.border_ratios[1])
        if self.scale:
            self.scaler_mean = data[:train_end].mean(axis=0)
            self.scaler_std = data[:train_end].std(axis=0)
            self.scaler_std[self.scaler_std == 0] = 1.0  # 避免除零
            self.data = (data - self.scaler_mean) / self.scaler_std
        else:
            self.data = data
            self.scaler_mean = np.zeros(data.shape[1])
            self.scaler_std = np.ones(data.shape[1])

        # 时间特征
        df_dates = pd.to_datetime(df['date'])
        dates = df_dates.values
        self.data_stamp = self._extract_time_features(dates)

        self.data = self.data[border1:border2]
        self.data_stamp = self.data_stamp[border1:border2]

        self.n_vars = self.data.shape[-1]
        self.text_embeds_slice = None
        self.img_tensors_slice = None

    def _extract_time_features(self, dates):
        """提取时间特征: [月, 日, 周几, 小时] → 归一化到 [0, 1]"""
        import warnings
        warnings.filterwarnings('ignore')
        df_stamp = pd.DataFrame({'date': dates})
        df_stamp['month'] = df_stamp['date'].dt.month / 12.0
        df_stamp['day'] = df_stamp['date'].dt.day / 31.0
        df_stamp['weekday'] = df_stamp['date'].dt.weekday / 6.0
        df_stamp['hour'] = df_stamp['date'].dt.hour / 23.0
        data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)
        return data_stamp

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]                  # [seq_len, C]
        seq_y = self.data[r_begin:r_end]                  # [label_len + pred_len, C]
        seq_x_mark = self.data_stamp[s_begin:s_end]       # [seq_len, 4]
        seq_y_mark = self.data_stamp[r_begin:r_end]       # [label_len + pred_len, 4]

        result = (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark),
        )

        # 多模态扩展
        if self.text_embeds_slice is not None:
            text = self.text_embeds_slice[s_begin:s_end].mean(axis=0)
            result = result + (torch.FloatTensor(text),)
        else:
            result = result + (torch.zeros(1),)  # placeholder

        # ★ v2.1: 卫星图像 (32×32 灰度图)
        if self.img_tensors_slice is not None:
            # img_tensors_slice: [N, 1, H, W]
            # 在窗口内对 H×W 图像平均 (与 text 一样)
            sat_imgs = self.img_tensors_slice[s_begin:s_end]  # [seq_len, 1, H, W]
            # 取窗口内平均 → [1, H, W] (保留空间信息)
            sat_avg = sat_imgs.mean(axis=0)
            result = result + (torch.FloatTensor(sat_avg),)
        else:
            result = result + (torch.zeros(1, 32, 32),)  # placeholder (32×32 灰度)

        return result

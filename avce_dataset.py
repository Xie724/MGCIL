import torch.utils.data as data
import numpy as np
from utils import process_feat, process_test_feat
import os

# 定义一个自定义数据集类，继承自 PyTorch 的 data.Dataset 类
class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False):
        """
        类的初始化方法，用于设置数据集的相关参数和解析数据列表。

        参数:
        args (Namespace): 包含各种配置参数的命名空间对象。
        transform (callable, 可选): 用于对数据进行转换的函数，默认为 None。
        test_mode (bool, 可选): 是否处于测试模式，默认为 False。
        """
        # 存储数据的模态信息，如 RGB、音频等
        self.modality = args.modality

        # 根据是否为测试模式，选择不同的 RGB 和音频数据列表文件
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list

        # 存储数据序列的最大长度
        self.max_seqlen = args.max_seqlen

        # 存储数据转换函数
        self.transform = transform

        # 标记当前是否处于测试模式
        self.test_mode = test_mode

        # 用于判断样本是否为正常样本的标志
        self.normal_flag = '_label_A'

        # 调用 _parse_list 方法解析数据列表文件
        self._parse_list()

    def _parse_list(self):
        """
        解析 RGB 和音频数据列表文件，将文件内容存储为列表。
        """
        # 打开 RGB 数据列表文件，并将每行内容作为一个元素存储在 self.list 中
        self.list = list(open(self.rgb_list_file))
        # 打开音频数据列表文件，并将每行内容作为一个元素存储在 self.audio_list 中
        self.audio_list = list(open(self.audio_list_file))

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本。

        参数:
        index (int): 样本的索引。

        返回:
        如果处于测试模式，返回 (f_v, f_a)，分别为视觉特征和音频特征。
        如果处于训练模式，返回 (f_v, f_a, label)，分别为视觉特征、音频特征和标签。
        """
        # 根据样本文件名是否包含正常样本标志，确定样本的标签
        if self.normal_flag in self.list[index]:
            label = 0.0  # 正常样本标签为 0
        else:
            label = 1.0  # 异常样本标签为 1

        # 从 RGB 数据列表中获取当前索引对应的文件路径，并加载视觉特征数据
        f_v = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        # 计算音频数据的索引，假设音频数据是每 5 个 RGB 样本对应一个音频样本
        audio_index = index // 5
        # 从音频数据列表中获取对应的文件路径，并加载音频特征数据
        f_a = np.array(np.load(self.audio_list[audio_index].strip('\n')), dtype=np.float32)

        # 如果定义了数据转换函数，对视觉和音频特征数据进行转换
        if self.transform is not None:
            f_v = self.transform(f_v)
            f_a = self.transform(f_a)

        # 根据是否处于测试模式，返回不同的数据组合
        if self.test_mode:
            return f_v, f_a
        else:
            # 对视觉和音频特征数据进行处理，使其长度符合最大序列长度要求
            f_v = process_feat(f_v, self.max_seqlen, is_random=False)
            f_a = process_feat(f_a, self.max_seqlen, is_random=False)
            return f_v, f_a, label

    def __len__(self):
        """
        返回数据集的样本数量。

        返回:
        int: 数据集的样本数量，即 RGB 数据列表的长度。
        """
        return len(self.list)
# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import time
import math

# 随机提取特征片段
# feat: 输入的特征数据，通常是一个二维数组
# t_max: 要提取的片段长度
# 该函数从输入特征数据中随机截取一段长度为 t_max 的片段
def random_extract(feat, t_max):
    # 生成一个随机整数 r，范围是从 0 到 len(feat) - t_max 之间
    r = np.random.randint(len(feat) - t_max)
    # 返回从 r 开始，长度为 t_max 的特征片段
    return feat[r:r + t_max]

# 均匀提取特征片段
# feat: 输入的特征数据，通常是一个二维数组
# t_max: 要提取的片段长度
# 该函数从输入特征数据中均匀采样 t_max 个点作为特征片段
def uniform_extract(feat, t_max):
    # 使用 np.linspace 生成 t_max 个在 0 到 len(feat) - 1 之间均匀分布的整数索引
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    # 根据生成的索引从 feat 中提取相应的特征片段
    return feat[r, :]

# 对特征数据进行填充
# feat: 输入的特征数据，通常是一个二维数组
# min_len: 最小长度，如果 feat 的长度小于 min_len 则进行填充
# 该函数将特征数据填充到指定的最小长度
def pad(feat, min_len):
    # 如果特征数据的第一维长度小于等于 min_len，则进行填充
    if np.shape(feat)[0] <= min_len:
        # 使用 np.pad 函数进行填充，在第一维填充，填充值为 0
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        # 如果特征数据的长度大于 min_len，则直接返回原特征数据
        return feat

# 处理特征数据
# feat: 输入的特征数据，通常是一个二维数组
# length: 目标长度
# is_random: 是否随机提取，默认为 True
# 该函数根据输入的特征数据和目标长度，对特征数据进行处理
def process_feat(feat, length, is_random=True):
    # 如果特征数据的长度大于目标长度
    if len(feat) > length:
        # 如果 is_random 为 True，则调用 random_extract 函数随机提取长度为 length 的片段
        if is_random:
            return random_extract(feat, length)
        else:
            # 如果 is_random 为 False，则调用 uniform_extract 函数均匀提取长度为 length 的片段
            return uniform_extract(feat, length)
    else:
        # 如果特征数据的长度小于等于目标长度，则调用 pad 函数进行填充
        return pad(feat, length)

# 处理测试特征数据
# feat: 输入的测试特征数据，通常是一个二维数组
# length: 目标长度
# 该函数根据输入的测试特征数据和目标长度，对测试特征数据进行处理
def process_test_feat(feat, length):
    # 记录输入特征数据的长度
    tem_len = len(feat)
    # 计算需要填充的倍数
    num = math.ceil(tem_len / length)
    # 如果特征数据的长度小于目标长度，则调用 pad 函数进行填充
    if len(feat) < length:
        return pad(feat, length)
    else:
        # 如果特征数据的长度大于等于目标长度，则填充到 num * length 的长度
        return pad(feat, num * length)

# 准备日志记录器
# eval: 是否为评估模式，默认为 False
# 该函数根据输入的评估模式，创建并配置一个日志记录器
def Prepare_logger(eval=False):
    # 获取当前模块的日志记录器
    logger = logging.getLogger(__name__)
    # 禁止日志记录器将日志消息传递给父记录器
    logger.propagate = False
    # 设置日志记录器的日志级别为 INFO
    logger.setLevel(logging.INFO)
    # 创建一个流处理器，用于将日志消息输出到控制台
    handler = logging.StreamHandler()
    # 创建一个格式化器，用于格式化日志消息
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # 将格式化器设置给流处理器
    handler.setFormatter(formatter)
    # 设置流处理器的日志级别为 0，表示处理所有级别的日志消息
    handler.setLevel(0)
    # 将流处理器添加到日志记录器中
    logger.addHandler(handler)

    # 获取当前时间的字符串表示，格式为 '%Y%m%d%H%M'
    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # 根据评估模式确定日志文件的名称
    logfile = 'log/' + date + '.log' if not eval else 'log/' + f'/{date}-Eval.log'
    # 确保日志目录存在（release 目录可能没有 log/）
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    # 创建一个文件处理器，用于将日志消息写入文件
    file_handler = logging.FileHandler(logfile, mode='w')
    # 设置文件处理器的日志级别为 INFO
    file_handler.setLevel(logging.INFO)
    # 创建一个格式化器，用于格式化日志消息
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # 将格式化器设置给文件处理器
    file_handler.setFormatter(formatter)
    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)

    # 返回配置好的日志记录器
    return logger

# 余弦退火调度器
# base_value: 初始值
# final_value: 最终值
# curr_epoch: 当前的 epoch 数
# epochs: 总的 epoch 数
# 该函数根据余弦退火算法计算当前 epoch 的值
def cosine_scheduler(base_value, final_value, curr_epoch, epochs):
    # 根据余弦退火算法计算当前 epoch 的值
    value = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * curr_epoch / epochs))
    return value
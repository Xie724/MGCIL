import numpy as np
import os
import cv2

# 定义每个视频片段的帧数
clip_len = 16

# 测试图像所在的目录，即测试视频的路径
video_root = '/cfs/cfs-4260a4096/417-mds2/justinjsyu/workspace/dance-music/XD/data/TestClips/Videos'

# 包含特征文件列表的文件路径
feature_list = 'rgb.list'

# 包含真实标签的文本文件路径
gt_txt = '/cfs/cfs-4260a4096/417-mds2/justinjsyu/workspace/dance-music/XD/data/annotations.txt'

# 读取真实标签文件的所有行
gt_lines = list(open(gt_txt))
# 用于存储所有视频的真实标签向量
gt = []

# 读取特征文件列表
lists = list(open(feature_list))
# 正常帧的总数量
tlens = 0
# 异常帧的总数量
vlens = 0

# 遍历特征文件列表中的每一项
for idx in range(len(lists)):
    # 去除每行末尾的换行符，并提取文件名
    name = lists[idx].strip('\n').split('/')[-1]
    # 只处理以 '__0.npy' 结尾的文件
    if '__0.npy' not in name:
        continue
    # 去掉文件名中的 '__0.npy' 后缀
    name = name[:-7]
    # 构建对应的视频文件名
    vname = name + '.mp4'
    # 打开视频文件
    cap = cv2.VideoCapture(os.path.join(video_root, vname))
    # 获取视频的总帧数
    lens = int(cap.get(7))

    # 初始化该视频的真实标签向量，初始值都为 0
    gt_vec = np.zeros(lens).astype(np.float32)
    # 如果文件名中不包含 '_label_A'
    if '_label_A' not in name:
        # 遍历真实标签文件的每一行
        for gt_line in gt_lines:
            # 如果当前视频名在该行中
            if name in gt_line:
                # 去除行末尾的换行符，并按空格分割
                gt_content = gt_line.strip('\n').split()
                # 提取异常片段的起始帧和结束帧
                abnormal_fragment = [[int(gt_content[i]), int(gt_content[j])] for i in range(1, len(gt_content), 2) \
                                     for j in range(2, len(gt_content), 2) if j == i + 1]
                # 如果存在异常片段
                if len(abnormal_fragment) != 0:
                    # 将异常片段转换为 numpy 数组
                    abnormal_fragment = np.array(abnormal_fragment)
                    # 遍历每个异常片段
                    for frag in abnormal_fragment:
                        # 将异常片段对应的帧的标签设为 1
                        gt_vec[frag[0]:frag[1]] = 1.0
                break

    # 计算视频帧数对 clip_len 取模的结果
    mod = (lens - 1) % clip_len  # 减 1 是为了与提取特征时的操作对齐
    # 去掉最后一帧
    gt_vec = gt_vec[:-1]
    # 如果取模结果不为 0
    if mod:
        # 去掉多余的帧
        gt_vec = gt_vec[:-mod]
    # 将该视频的真实标签向量添加到总标签列表中
    gt.extend(gt_vec)
    # 如果该视频存在异常帧
    if sum(gt_vec) / len(gt_vec):
        # 累加正常帧的数量
        tlens += len(gt_vec)
        # 累加异常帧的数量
        vlens += sum(gt_vec)

# 将所有视频的真实标签向量保存为 numpy 文件
np.save('gt.npy', gt)
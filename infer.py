# 从torch.utils.data中导入DataLoader，用于创建数据加载器
from torch.utils.data import DataLoader
# 导入torch库，用于深度学习相关操作
import torch
# 导入numpy库，用于数值计算
import numpy as np
# 从avce_network模块中导入AVCE_Model并将其重命名为Model，这是要使用的模型类
from avce_network import AVCE_Model as Model
# 从avce_dataset模块中导入Dataset类，用于加载数据集
from avce_dataset import Dataset
# 从test模块中导入avce_test函数并将其重命名为test，用于进行模型测试
from test import avce_test as test
# 导入option模块，用于解析命令行参数
import option
# 导入time模块，用于计时
import time

# 主程序入口
if __name__ == '__main__':
    # 解析命令行参数，将解析结果存储在args中
    args = option.parser.parse_args()

    # 创建一个数据加载器test_loader，用于加载测试数据集
    # Dataset(args, test_mode=True) 创建一个测试模式的数据集对象
    # batch_size=5 表示每次从数据集中加载5个样本
    # shuffle=False 表示不打乱数据顺序
    # num_workers=args.workers 表示使用args.workers个线程来加载数据
    # pin_memory=True 表示将数据加载到固定内存中，加速数据传输到GPU
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # 创建模型实例，使用解析的参数args进行初始化
    model = Model(args)
    # 将模型移动到GPU上进行计算
    model = model.cuda()

    # 加载预训练的模型参数
    # torch.load(args.model_dir) 加载存储在args.model_dir路径下的模型参数
    # {k.replace('module.', ''): v for k, v in ...} 去除参数名中的'module.'前缀
    # model.load_state_dict(...) 将处理后的参数加载到模型中
    # model_dict = model.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load(args.model_dir).items()})
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(args.model_dir).items()},
        strict=False  # <--- 新增：允许权重不完全匹配，防止加载旧模型时报错
    )

    # 加载真实标签数据，存储在gt中
    gt = np.load(args.gt)

    # 记录开始时间
    st = time.time()

    # 调用test函数进行模型测试，传入测试数据加载器、模型、None（可能表示不使用某些额外模型）、真实标签和0（可能是一个epoch编号）
    # 函数返回av_ap和另一个值（这里用_表示忽略，单模态分支AP）

    av_ap, _ = test(test_loader, model, None, gt, 0, use_gts=bool(args.use_gts), sigma=args.gts_sigma)
    # 计算并打印测试所花费的时间
    print('Time:{}'.format(time.time() - st))
    # 打印离线av_ap指标，保留4位小数
    print('offline av_ap:{:.4f} (gts={}, sigma={})'.format(av_ap, args.use_gts, args.gts_sigma))

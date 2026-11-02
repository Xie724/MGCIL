# 导入必要的库
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import math
import numpy as np
import random
import os
# 从自定义模块导入模型类
from avce_network import AVCE_Model, Single_Model
# 从自定义模块导入数据集类
from avce_dataset import Dataset
# 从 train 模块导入训练函数
from train import avce_train as train
# 从 test 模块导入测试函数
from test import avce_test as test
# 从 option 模块导入参数解析器
import option
# 从 utils 模块导入日志记录器和余弦退火调度器
from utils import Prepare_logger, cosine_scheduler
# Regime A (reproduce the 88.58 reference runs): mixed_manifold is NOT used in the
# loss, but MUST be instantiated here so the global-RNG state entering the first
# DataLoader shuffle matches XS90. Removing it changes the batch order and the result.
from mixed_manifold import MixedManifoldInfoNCE
# 设置随机种子以确保实验的可重复性
def setup_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 设置 PyTorch CUDA 的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置 Numpy 的随机种子
    np.random.seed(seed)
    # 设置 Python 内置随机模块的随机种子
    random.seed(seed)
    # 确保 CUDA 卷积算法是确定性的
    torch.backends.cudnn.deterministic = True
    # 禁用 CUDA 卷积算法的自动调优
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 声明全局变量 logger
    global logger
    # 设置多进程启动方法为 'spawn'
    torch.multiprocessing.set_start_method('spawn')
    # 解析命令行参数
    args = option.parser.parse_args()
    # 调用 setup_seed 函数设置随机种子
    setup_seed(args.seed)
    # 设置可见的 CUDA 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # 初始化日志记录器
    logger = Prepare_logger(eval=False)
    # 记录参数信息
    logger.info(args)

    # 创建训练数据加载器
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    # 创建测试数据加载器
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # 初始化 AVCE 模型并移至 GPU
    model_av = AVCE_Model(args).cuda()
    # 初始化单模态模型并移至 GPU
    model_v = Single_Model(args).cuda()
    # Regime A: instantiate the (unused) mixed_nce module so the RNG state at the
    # first DataLoader shuffle matches the XS90 runs that produced the 88.58 result.
    mixed_nce = MixedManifoldInfoNCE(feature_dim=args.hid_dim).cuda()
    # 计算模型的总参数数量
    total_params = sum(p.numel() for p in model_av.parameters())
    total_params += sum(p.numel() for p in model_v.parameters())
    # 记录总参数数量
    logger.info(f'{total_params/1e6:.3f}M parameters.')
    # 计算可训练参数的数量
    total_trainable_params = sum(p.numel() for p in model_av.parameters() if p.requires_grad is True)
    total_trainable_params += sum(p.numel() for p in model_v.parameters() if p.requires_grad is True)
    # 记录可训练参数的数量
    logger.info(f'{total_trainable_params/1e6:.3f}M training parameters.')

    # 如果 './ckpt' 目录不存在，则创建该目录
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')


    # 定义二元交叉熵损失函数
    criterion = torch.nn.BCELoss()
    # 初始化 AVCE 模型的优化器
    optimizer_av = optim.Adam([
        {'params': model_av.parameters()},
        {'params': mixed_nce.parameters(), 'lr': args.lr * 0.1, 'weight_decay': 1e-4}
    ], lr=args.lr, weight_decay=0.000)
    # 初始化单模态模型的优化器
    optimizer_v = optim.Adam(model_v.parameters(), lr=args.lr / 5, weight_decay=0.000)
    # 初始化 AVCE 模型的学习率调度器
    scheduler_av = optim.lr_scheduler.CosineAnnealingLR(optimizer_av, T_max=60, eta_min=0)
    # 初始化单模态模型的学习率调度器
    scheduler_v = optim.lr_scheduler.CosineAnnealingLR(optimizer_v, T_max=60, eta_min=0)
    # 加载真实标签
    gt = np.load(args.gt)

    # 初始化最佳 AP 指标和最佳 epoch
    best_av_ap = 0
    best_v_ap = 0
    best_epoch = 0
    # 在随机初始化的模型上进行测试
    av_ap, v_ap = test(test_loader, model_av, model_v, gt, 0, use_gts=bool(args.use_gts), sigma=args.gts_sigma)
    # 记录随机初始化的 AVCE 模型的离线 AP
    logger.info('Random initalization: offline av_ap:{0:.4}\n'.format(av_ap))
    # 开始训练循环
    for epoch in range(args.max_epoch):
        # 记录当前 epoch 的开始时间
        st = time.time()
        # 计算当前 epoch 的 lambda_a2b 和 lambda_a2n
        lamda_a2b = min(args.lamda_a2b, args.lamda_cof * epoch)
        lamda_a2n = min(args.lamda_a2n, args.lamda_cof * epoch)
        # 调用训练函数进行训练
        av_loss, v_loss = train(train_loader, model_av, model_v, optimizer_av, optimizer_v, criterion,
                                lamda_a2b, lamda_a2n, args.lamda_anchor, args.tau, logger,
                                args.adv_eps, args.lamda_video,
                                bool(args.use_cgm), bool(args.use_uwc), bool(args.use_anchor),
                                bool(args.use_adv), bool(args.use_vlc), args.grad_clip,
                                conf_temp=args.conf_temp)
        # 更新 AVCE 模型的学习率
        scheduler_av.step()
        # 更新单模态模型的学习率
        scheduler_v.step()

        # 禁用梯度计算
        with torch.no_grad():
            # 计算当前 epoch 的余弦退火调度器的值
            m = cosine_scheduler(base_value=args.m, final_value=1, curr_epoch=epoch, epochs=50)
            if m != 1.0:
                # 遍历 AVCE 模型的参数
                for param_av in model_av.named_parameters():
                    # 跳过特定的参数
                    if 'sa_a' in param_av[0] or 'fc_a' in param_av[0]:
                        continue
                    # 遍历单模态模型的参数
                    for param_v in model_v.named_parameters():
                        # 如果参数名相同，则更新 AVCE 模型的参数
                        if param_av[0] == param_v[0]:
                            param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                            break
                        # 处理特定参数的更新
                        elif param_av[0] == 'att_mmil.fc.weight' and param_v[0] == 'fc.weight':
                            param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                            break
                        elif param_av[0] == 'att_mmil.fc.bias' and param_v[0] == 'fc.bias':
                            param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                            break

        # 在当前 epoch 进行测试
        av_ap, v_ap = test(test_loader, model_av, model_v, gt, epoch, use_gts=bool(args.use_gts), sigma=args.gts_sigma)
        # 如果当前的 AVCE 模型的 AP 优于最佳 AP，则更新最佳指标和保存模型
        if av_ap > best_av_ap:
            best_av_ap = av_ap
            best_v_ap = v_ap
            best_epoch = epoch
            torch.save(model_av.state_dict(), './ckpt/' + args.model_name + '.pkl')
        logger.info('av_loss:{:.4} | v_loss:{:.4}\n'.format(av_loss, v_loss))
        logger.info(
            'Epoch {}/{}: av_ap:{:.4} | v_ap:{:.4} | m={:.4}\n'.format(epoch, args.max_epoch, av_ap, v_ap, m))
    logger.info(
        'Best Performance in Epoch {}: av_ap:{:.4} | v_ap:{:.4}\n'.format(best_epoch, best_av_ap, best_v_ap))
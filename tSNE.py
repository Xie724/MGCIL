#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.

import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch


# torch.set_default_tensor_type(torch.cuda.DoubleTensor)

# 计算给定距离矩阵 D 和精度参数 beta 下的高斯核概率分布 P 以及熵 H
def Hbeta_torch(D, beta=1.0):
    # 计算高斯核概率分布 P
    P = torch.exp(-D.clone() * beta)
    # 计算 P 的总和
    sumP = torch.sum(P)
    # 计算熵 H
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    # 归一化 P
    P = P / sumP
    return H, P


# 计算相似度矩阵 P，通过二分搜索为每个数据点找到合适的精度参数 beta，使得每个条件高斯分布的困惑度相同
def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    # 初始化一些变量
    print("Computing pairwise distances...")
    # 获取数据点的数量 n 和特征维度 d
    (n, d) = X.shape
    # 计算每个数据点的平方和
    sum_X = torch.sum(X * X, 1)
    # 计算距离矩阵 D
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)
    # 初始化相似度矩阵 P
    P = torch.zeros(n, n)
    # 初始化精度参数 beta
    beta = torch.ones(n, 1)
    # 计算困惑度的对数
    logU = torch.log(torch.tensor([perplexity]))
    # 数据点的索引列表
    n_list = [i for i in range(n)]

    # 遍历所有数据点
    for i in range(n):
        # 打印进度信息
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))
        # 初始化二分搜索的上下界
        betamin = None
        betamax = None
        # 获取第 i 个数据点与其他数据点的距离
        Di = D[i, n_list[0:i] + n_list[i + 1:n]]
        # 计算当前精度下的熵和概率分布
        (H, thisP) = Hbeta_torch(Di, beta[i])
        # 计算熵与目标困惑度的差值
        Hdiff = H - logU
        tries = 0
        # 二分搜索，调整 beta 使得熵接近目标困惑度
        while torch.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                # 熵过大，增大 beta
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                # 熵过小，减小 beta
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            # 重新计算熵和概率分布
            (H, thisP) = Hbeta_torch(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        # 设置相似度矩阵 P 的第 i 行
        P[i, n_list[0:i] + n_list[i + 1:n]] = thisP

    # 返回最终的相似度矩阵 P
    return P


# 使用 PCA 对数据进行预处理
def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    # 获取数据点的数量 n 和特征维度 d
    (n, d) = X.shape
    # 数据中心化
    X = X - torch.mean(X, 0)
    # 计算协方差矩阵的特征值和特征向量
    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # 处理复特征值，将复特征值对应的特征向量替换为实部
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i + 1] = M[:, i]
            i += 2
        else:
            i += 1
    # 降维到 no_dims 维
    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


# 执行 t-SNE 算法
def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    # 设置默认的张量类型为 DoubleTensor
    torch.set_default_tensor_type(torch.DoubleTensor)
    # 检查输入
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # 初始化变量
    # 使用 PCA 对数据进行降维
    X = pca_torch(X, initial_dims)
    # 获取数据点的数量 n 和特征维度 d
    (n, d) = X.shape
    # 最大迭代次数
    max_iter = 1000
    # 初始动量
    initial_momentum = 0.5
    # 最终动量
    final_momentum = 0.8
    # 学习率
    eta = 500
    # 最小增益
    min_gain = 0.01
    # 随机初始化低维嵌入 Y
    Y = torch.randn(n, no_dims)
    # 初始化速度 dY
    dY = torch.zeros(n, no_dims)
    # 初始化动量 iY
    iY = torch.zeros(n, no_dims)
    # 初始化增益 gains
    gains = torch.ones(n, no_dims)

    # 计算相似度矩阵 P
    P = x2p_torch(X, 1e-5, perplexity)
    # 对称化 P
    P = P + P.t()
    # 归一化 P
    P = P / torch.sum(P)
    # 早期夸张
    P = P * 4.
    print("get P shape", P.shape)
    # 避免出现零值
    P = torch.max(P, torch.tensor([1e-21]))

    # 迭代优化
    for iter in range(max_iter):
        # 计算低维空间的相似度矩阵 Q
        sum_Y = torch.sum(Y * Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        # 避免出现零值
        Q = torch.max(Q, torch.tensor([1e-12]))
        # 计算梯度
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # 执行更新
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        # 更新增益
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        # 更新动量
        iY = momentum * iY - eta * (gains * dY)
        # 更新低维嵌入 Y
        Y = Y + iY
        # 中心化 Y
        Y = Y - torch.mean(Y, 0)

        # 每 10 次迭代计算一次损失
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # 100 次迭代后取消早期夸张
        if iter == 100:
            P = P / 4.

    # 返回低维嵌入结果
    return Y


# 批量执行 t-SNE 并绘制可视化结果
def batch_tsne(feat, labels, epoch, num, dir):
    # 确保特征和标签的长度一致
    assert (len(feat) == len(labels))
    # 将特征转换为张量
    feat = torch.Tensor(feat)
    # 禁用梯度计算
    with torch.no_grad():
        # 执行 t-SNE 算法
        Y = tsne(feat, 2, 128, 20.0)

    # 将结果转换为 numpy 数组
    Y = Y.cpu().numpy()
    # 绘制散点图
    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    # 获取当前坐标轴
    ax = pyplot.gca()
    # 隐藏 x 轴和 y 轴
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # 显示图像
    pyplot.show()
    # 保存图像
    pyplot.savefig(dir + str(epoch) + '_' + str(num) + '.png')
    # 清除当前图像
    pyplot.clf()


if __name__ == "__main__":
    # 读取特征文件列表
    rgb_list = list(open('list/rgb.list'))
    # 加载标签
    total_labels = np.load('list/gt.npy')
    # 加载第一个特征文件
    X = np.array(np.load(rgb_list[0].strip('\n')), dtype=np.float32)
    # 将特征转换为张量
    X = torch.Tensor(X)
    # 随机生成标签
    labels = np.random.randint(0, 2, (X.size(0)))
    # 确认特征和标签的长度一致
    assert (len(X[:, 0]) == len(X[:, 1]))
    assert (len(X) == len(labels))

    # 禁用梯度计算
    with torch.no_grad():
        # 执行 t-SNE 算法
        Y = tsne(X, 2, 50, 20.0)
    # 将结果转换为 numpy 数组
    Y = Y.cpu().numpy()
    # 绘制散点图
    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    # 保存图像
    pyplot.savefig('tSNE.png')
    # 清除当前图像
    pyplot.clf()
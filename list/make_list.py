import os
import glob

# 定义特征文件所在的根目录路径
root_path = '/home/stu2023/xs/data/audiovideo/RGB/'    ## the path of features

# 使用 glob.glob 函数获取指定根目录下所有以 .npy 结尾的文件的路径，并按文件名排序
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

# 打印找到的 .npy 文件的数量
print(len(files))

# 用于存储暴力相关特征文件路径的列表
violents = []
# 用于存储正常相关特征文件路径的列表
normal = []

# 以读写模式打开一个名为 rgb.list 的文件，如果文件不存在则创建它
with open('/home/stu2023/xs/project/MACIL_SD-main/list/rgb.list', 'w+') as f:  ## the name of feature list
    # 遍历所有找到的 .npy 文件
    for file in files:
        # 检查文件名中是否包含 '_label_A'
        if '_label_A' in file:
            # 如果包含，则认为是正常相关的文件，将其路径添加到 normal 列表中
            normal.append(file)
        else:
            # 如果不包含，则认为是暴力相关的文件，在文件名后添加换行符
            newline = file + '\n'
            # 将该文件路径写入 rgb.list 文件中
            f.write(newline)
    # 遍历正常相关文件的列表
    for file in normal:
        # 在文件名后添加换行符
        newline = file + '\n'
        # 将正常相关文件的路径写入 rgb.list 文件中
        f.write(newline)
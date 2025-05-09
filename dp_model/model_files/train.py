from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torchvision import transforms
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast


# 数据加载类
class MRIDataset(Dataset):
    def __init__(self, image_dir, age_dir, image_paths, labels):               ####labels
        self.image_dir = image_dir
        self.age_dir = age_dir
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_name)
        age_path = os.path.join(self.age_dir, image_name)  # 假设图像和年龄文件名相同

        # 加载图像和对应的年龄
        img = np.load(image_path)
        label = np.load(age_path)  # 加载与图像对应的年龄标签

        # 标准化：将图像归一化到 [0, 1] 范围
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # 扩展维度：变为 (1, D, H, W)
        img = np.expand_dims(img, axis=0)

        # 转换为 PyTorch 张量
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label


# 加载数据
def load_data(mri_dir, age_dir):
    # 获取文件夹中的所有文件名
    image_names = [f for f in os.listdir(mri_dir) if f.endswith('.npy')]  # 读取.npy文件

    print(f"Total MRI images found: {len(image_names)}")

    image_paths = []
    labels = []

    # 遍历每个文件，读取对应的年龄标签
    for image_name in image_names:
        image_paths.append(image_name)

        # 假设图像文件名和标签文件名相同，标签存储在 AGE 文件夹中
        age_path = os.path.join(age_dir, image_name)

        # 加载年龄标签，转换成软标签
        label = np.load(age_path)
        bin_range = [54,96]
        bin_step = 1
        sigma = 1
        y, bc = dpu.num2vect(label, bin_range, bin_step, sigma)
        y = torch.tensor(y, dtype=torch.float32)
        labels.append(y)

    return image_paths, labels


# 设置参数
mri_dir = '/data2/wangchangmiao/liuxiaoshuai/IXI/T1'   # MRI 图像所在目录
age_dir = '/data2/wangchangmiao/liuxiaoshuai/IXI/Age'  # 年龄标签所在目录

image_paths, labels = load_data(mri_dir, age_dir)

# 将数据集划分为训练集和验证集
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.3, random_state=42)

# 创建数据集和数据加载器
train_dataset = MRIDataset(mri_dir, age_dir, train_paths, train_labels)
val_dataset = MRIDataset(mri_dir, age_dir, val_paths, val_labels)
print(len(train_dataset))
print(len(val_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24, shuffle=False)


# 实例化模型

model = SFCN(output_dim=40)              ## IXI数据集20-86岁，ADNI总体上55-95
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 设置损失函数和优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,  # 设置学习率为 0.0001
    betas=(0.9, 0.999)  # 设置 beta1=0.9, beta2=0.999
)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练循环

num_epochs = 50

# 在训练循环中，收集预测结果和真实标签
train_predictions = []
train_true_labels = []
train_image_names = []

# 在验证循环中，收集预测结果和真实标签
val_predictions = []
val_true_labels = []
val_image_names = []
scaler = GradScaler()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, y) in enumerate(train_loader):
        inputs, y = inputs.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():  # 自动混合精度
            outputs = model(inputs)
            x = outputs
            loss = dpl.my_KLDivLoss(x, y)                                        ###输出MSE,MAE

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #print(outputs.shape)
        #print(targets.shape)
        #loss.backward()
        #optimizer.step()

        running_loss += loss.item()                      

    if epoch == num_epochs - 1:
        # 收集预测结果和真实标签
        train_predictions.extend(outputs.cpu().numpy().reshape(-1, 40))  # 对应 one-hot 40 维度
        train_true_labels.extend(y.cpu().numpy().reshape(-1, 40))
        #train_predictions.extend(outputs.detach().cpu().numpy())
        #train_true_labels.extend(targets.cpu().numpy())

        #train_image_names.extend(train_loader.dataset.image_paths[i * train_loader.batch_size:(i + 1) * train_loader.batch_size])
        batch_image_names = train_loader.dataset.image_paths[
                            i * train_loader.batch_size:(i + 1) * train_loader.batch_size]
        train_image_names.extend(batch_image_names)


        train_results = pd.DataFrame({
        'Image Name': train_image_names,
        'True Age': train_true_labels,
        'Predicted Age': train_predictions
        })

        train_results.to_csv('train_results.csv', index=False)
        print("训练结果已保存到 'train_results.csv'")

    # 输出训练损失
    print("训练集:")
    print(f"Epoch [{epoch + 1}/{num_epochs}], MSE: {running_loss / len(train_dataset)}")

    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_loss_mae = 0.0
        for inputs, y in val_loader:
            inputs, y = inputs.to(device), y.to(device)

            outputs = model(inputs)
            x = outputs
            loss = dpl.my_KLDivLoss(x, y) 
            val_loss += loss.item()

            
            '''
            print("预测值:")
            print(outputs)
            print("真实值:")
            print(targets)
            '''

        
        if epoch == num_epochs - 1:
            # 收集预测结果和真实标签
            #val_predictions.append(outputs)
            #val_true_labels.append(targets)
            val_predictions.extend(outputs.cpu().numpy())
            val_true_labels.extend(targets.cpu().numpy())
            #val_image_names.extend(val_loader.dataset.image_paths[i * val_loader.batch_size:(i + 1) * val_loader.batch_size])

            batch_image_names = val_loader.dataset.image_paths[
                                i * val_loader.batch_size:(i + 1) * val_loader.batch_size]
            val_image_names.extend(batch_image_names)

            print("验证结果:")
            print(f'True:{val_true_labels}')
            print(f'Predicted:{val_predictions}')
        
        
        # 输出测试损失
        print("测试集:")
        print(f"Epoch [{epoch + 1}/{num_epochs}], MSE: {val_loss / len(val_dataset)}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], MAE: {val_loss_mae / len(val_dataset)}")

# 在最后一次 epoch 后，将结果保存到 Excel 文件


'''
print(len(val_image_names))
print(len(val_true_labels))
print(len(val_predictions))
# 将验证结果保存到另一个 Excel 文件
val_results = pd.DataFrame({
    'Image Name': val_image_names,
    'True Age': val_true_labels,
    'Predicted Age': val_predictions
})
val_results.to_csv('val_results.csv', index=False)
print("验证结果已保存到 'val_results.csv'")
'''

# 最后保存模型
model_save_path = f'brain_age_model_{time.strftime("%Y%m%d_%H%M%S")}.pth'
torch.save(model.state_dict(), model_save_pth)

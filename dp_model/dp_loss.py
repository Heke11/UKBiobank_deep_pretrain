import torch.nn as nn

def my_KLDivLoss(x, y):            # x,y都是[batch_size, 40]
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]            # n是批量大小
    loss = loss_func(x, y) / n
    #print(loss)
    return loss            # 一个批量中每个样本的平均损失

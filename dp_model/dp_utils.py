import numpy as np
from scipy.stats import norm

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:               
        '''
        用于性别分类
        sigma = 0 实现的是硬标签（hard label），对应于硬分类的情况，即每个数据点只属于一个明确的类别
        '''
        x = np.array(x)
        # a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        # print(np.floor(a)) # Output: [-2. -2. -1. 0. 1. 1. 2.]   np.floor()转换成小于原来的数的最大整数
        i = np.floor((x - bin_start) / bin_step)                
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:           
        '''
        用于年龄回归
        当 sigma > 0 时，标签不是一个硬分类（hard label），而是一个概率分布。
        这个概率分布以真实年龄为中心，并根据sigma决定分布的宽度。
        这种方法通过计算预测的概率分布与真实的软标签分布之间的KL散度（Kullback-Leibler divergence）来计算损失
        '''
        if np.isscalar(x):                             # np.isscaler(x)判断x是否是标量类型，标量就是只有单个值，数组不是标量
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)            # norm.cdf()是把样本年龄映射到一个概率分布，cdf计算累积概率
                v[i] = cdfs[1] - cdfs[0]                # cdfs[0]是累积到x1的概率，cdfs[1]是累积到x2的概率，v[i]是区间[x1,x2]之间的概率
            return v, bin_centers            # v包含了这个样本年龄对应的所有年龄区间的概率分布
        else:            # x包含多个样本年龄
            v = np.zeros((len(x), bin_number))        
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        
        
def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

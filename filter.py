import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from IPython import embed

featNme=['BengChukuo','FaJin','TuiYouGangJin','TuiYouWangChu','MaDajin',
         'MaDaChu','JiaYouGangJin','JiaYouWangChu','JiGuang','NiuJu','ZhuanSu']
def wiener(Allata,isShow=False):
    "是signal模块中的滤波函数，其输入参数分别是待滤波数据和滤波模板"
    CopyData = Allata.reshape((-1, Allata.shape[-1]))
    x = np.linspace(0, 1, CopyData.shape[0])
    ret=[]
    for i in range(CopyData.shape[-1]):
        w = ss.wiener(CopyData[:,i], 10)  # 维纳滤波
        ret.append(w)
        if isShow:
            plt.plot(x, CopyData[:,i], c='b', label=featNme[i]+' src')
            plt.plot(x, w, c = 'r', label=featNme[i])
            plt.legend()
            plt.show()

    return np.array(ret).reshape(Allata.shape[0],Allata.shape[1],Allata.shape[-1])

def butter(Allata,isShow=False):
    """
    巴特沃斯滤波器
    :param Allata:
    :param isShow:
    :return:
    """
    CopyData = Allata.reshape((-1, Allata.shape[-1]))
    x = np.linspace(0, 1, CopyData.shape[0])
    ret = []
    for i in range(CopyData.shape[-1]):
        b, a = ss.butter(3, 0.05)
        w = ss.lfilter(b, a, CopyData[:,i])
        if isShow:
            plt.plot(x, CopyData[:, i], c='b', label=featNme[i] + ' src')
            plt.plot(x, w, c='r', label=featNme[i])
            plt.legend()
            plt.show()
    return np.array(ret).reshape(Allata.shape[0], Allata.shape[1], Allata.shape[-1])









# PI = 2*np.pi
# x = (np.sin(PI*0.75*t*(1-t) + 2.1) +
#      0.1*np.sin(PI*1.25*t + 1) +
#      0.18*np.cos(PI*3.85*t))
#
# # 原始数据添加噪声
# np.random.seed(42)
# xn = x + np.random.rand(len(t))
#
#
#


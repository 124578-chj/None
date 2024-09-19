import os
import pandas as pd
from IPython import embed
import numpy as np
# 1泵出口 2多路阀进	3推进油缸进	4推进油缸出	5马达进	6马达出	7加载油缸进	8加载油缸出 9激光位移	10扭矩	11转速
fratNme=['BengChukuo','FaJin','TuiYouGangJin','TuiYouWangChu','MaDajin',
         'MaDaChu','JiaYouGangJin','JiaYouWangChu','JiGuang','NiuJu','ZhuanSu']
Names=['hemei','niye','shaye','wuyanmei','yanmei','yeyan']

if __name__ == '__main__':
    path=r'C:\Users\HP\Desktop\0918\Dataset'
    freq=50 #采样频率
    LenOne=100 #单个样本取多少行数据


    label = []
    TData = []
    for filename in Names:
        csvPath=os.path.join(path,filename)
        for csvname in os.listdir(csvPath):
            if csvname.endswith('.csv'):
                print(os.path.join(csvPath,csvname))
                csvData= pd.read_csv(os.path.join(csvPath,csvname))
                Tempnum=int(csvData['Time'].shape[0]/freq)
                j=0
                singlelbael = Names.index(filename)
                for j in range(freq):
                    TempData = []
                    for i in range(Tempnum-1): #按照freq
                        index=j+freq*i
                        rowData=csvData.loc[index][1:]  #获取11个特征
                        TempData.append(rowData)
                    j+=1
                    TData.append(np.array(TempData[:LenOne]))
                    np.array(TData)
                    label.append(singlelbael)

    npLabel=np.array(label)    #保存每个样本 n,1
    npTata=np.array(TData)  #保存抽取的特征 n,100,11

    np.save(os.path.join("Dataset","label.npy"),npLabel)
    np.save(os.path.join("Dataset","Data.npy"),npTata)





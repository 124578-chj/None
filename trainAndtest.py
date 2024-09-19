from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
from filter import *
if __name__ == '__main__':
    #加载数据集
    Allata=np.load(os.path.join("Dataset","Data.npy"))
    Alllabel=np.load(os.path.join("Dataset","label.npy"))

    Retfilter=butter(Allata,True)

    #特征归一化




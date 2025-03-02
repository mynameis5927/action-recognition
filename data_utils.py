import torch
from os import listdir
from os import chdir
from os import getcwd
from os.path import join
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

class Joint:
    HipCenter = 6
    Spine = 3
    ShoulderCenter = 2
    Head = 19
    ShoulderLeft = 1
    ElbowLeft = 8
    WristLeft = 10
    HandLeft = 12
    ShoulderRight = 0
    ElbowRight = 7
    WristRight = 9
    HandRight = 11
    HipLeft = 5
    KneeLeft = 14
    AnkleLeft = 16
    FootLeft = 18
    HipRight = 4
    KneeRight = 13
    AnkleRight = 15
    FootRight = 17

edges_between_joints = (
    (Joint.ShoulderCenter,Joint.ShoulderLeft),
    (Joint.ShoulderLeft, Joint.ElbowLeft),
    (Joint.ElbowLeft, Joint.WristLeft),
    (Joint.WristLeft, Joint.HandLeft),

    (Joint.ShoulderCenter,Joint.ShoulderRight),
    (Joint.ShoulderRight,Joint.ElbowRight),
    (Joint.ElbowRight,Joint.WristRight),
    (Joint.WristRight,Joint.HandRight),

    (Joint.Head,Joint.ShoulderCenter),
    (Joint.ShoulderCenter, Joint.Spine),
    (Joint.Spine, Joint.HipCenter),

    (Joint.HipCenter, Joint.HipRight),
    (Joint.HipRight, Joint.KneeRight),
    (Joint.KneeRight, Joint.AnkleRight),
    (Joint.AnkleRight, Joint.FootRight),

    (Joint.HipCenter, Joint.HipLeft),
    (Joint.HipLeft, Joint.KneeLeft),
    (Joint.KneeLeft, Joint.AnkleLeft),
    (Joint.AnkleLeft, Joint.FootLeft),
    )

# This function can read test, train and validation data, you need to call it for each of them seper.
def read_msrdataset(data_dir,timesteps,normalize=True):
    print('Loading MSR 3D Data, data directory %s' % data_dir)
    numOfJoints = 20
    maxValue = 3.879377  # dataset attribute 
    minValue = -1.878035 # dataset attribute
    frameSeqs = 25000 # first, allocate more than needed, after reading, delete unnecessary allocation

    prevDir = getcwd()
    chdir(data_dir)
    documents = [d for d in sorted(listdir("."))]
    
    inpData = np.zeros((timesteps,frameSeqs,numOfJoints*3), dtype=np.float32)
    labels = np.zeros((frameSeqs), dtype=np.int64)
    batchLens = np.zeros((len(documents),2), dtype=np.int64)
    trainPrevActionIdx = 0
    for fIdx,file in enumerate(documents):
        currentLabel = int(file[1:3]) 
        action = np.loadtxt(file)
        action = np.delete(action, 3, axis=1) # delete the unnecessary last column
        numOfFrames = action.shape[0] // numOfJoints
        action = np.reshape(action,( numOfFrames, numOfJoints*3 ))
        if normalize:
            # Frame normalization
            spineCoordinates = action[:,Joint.Spine*3:Joint.Spine*3+3]
            hipCenterCoordinates = action[:,Joint.HipCenter*3:Joint.HipCenter*3+3]
            for i in range(0,60,3):
                action[:,i:i+3] -= (spineCoordinates+hipCenterCoordinates)/2
            # Dataset normalization
            action=(action+abs(minValue))/(maxValue+abs(minValue))

        zeroPaddedAction = np.concatenate((np.zeros((timesteps-1, numOfJoints*3)), action))
        batchLens[fIdx] = [trainPrevActionIdx, numOfFrames-1+trainPrevActionIdx]
        bs, be = batchLens[fIdx]
        
        for step in range(timesteps):
            inpData[step,bs:be] = zeroPaddedAction[step:step+numOfFrames-1]
        labels[bs:be] = currentLabel-1
        trainPrevActionIdx = numOfFrames-1+trainPrevActionIdx

    # free unnecessary allocation
    inpData = np.delete(inpData, range(batchLens[-1][1],frameSeqs) , axis=1)
    labels = np.delete(labels, range(batchLens[-1][1],frameSeqs), axis=0)
    chdir(prevDir)

    return inpData,labels

# 假设数据已加载为numpy数组，形状如下：
# 训练集：(40, 18329, 60) → (时间步, 样本数, 特征)
# 测试集：(40, 4582, 60)
# 转换为PyTorch张量并进行维度调整
def load_data():

    trainDir = "./MSRAction3DSkeletonReal3D" 
    trainData, trainLabels = read_msrdataset(trainDir,40)
    tLEn = trainData.shape[1]
    p = np.random.permutation(trainData.shape[1])
    trainData, trainLabels = trainData[:,p,:],trainLabels[p]
    
    test_data, test_labels = trainData[:,(tLEn-tLEn//5):,:],trainLabels[(tLEn-tLEn//5):]
    train_data, train_labels = trainData[:,:(tLEn-tLEn//5),:],trainLabels[:(tLEn-tLEn//5)]
    
    # 示例数据生成（替换为实际数据）
    # train_data = np.random.randn(40, 18329, 60).astype(np.float32)
    # test_data = np.random.randn(40, 4582, 60).astype(np.float32)
    # train_labels = np.random.randint(0, 20, (18329,))  # 假设20个动作类别
    # test_labels = np.random.randint(0, 20, (4582,))
 
    # 维度转换：(时间步, 样本数, 特征) → (样本数, 时间步, 特征)
    train_data = torch.tensor(train_data.transpose(1, 0, 2))
    test_data = torch.tensor(test_data.transpose(1, 0, 2))
    
    # 转换为骨架格式：(样本数, 时间步, 节点数, 坐标)
    train_data = train_data.view(-1, 40, 20, 3)  # 60维特征分解为20节点×3坐标
    test_data = test_data.view(-1, 40, 20, 3)
    
    # 创建数据集
    train_dataset = TensorDataset(train_data, torch.tensor(train_labels))
    test_dataset = TensorDataset(test_data, torch.tensor(test_labels))
    return train_dataset, test_dataset

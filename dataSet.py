# import h5py
# import numpy as np
# import torch.utils.data as data
# import glob
#
# parentNum=4
# PLY_EXTENSION=['MPEG','MVUB']
# def is_ply_file(filename):
#     return any(extension in filename for extension in PLY_EXTENSION)
#
# #加载点云数据
# def plyLoader(filePath):
#     mat=h5py.File(filePath)
#     cell=mat['patchFile']
#     return cell,mat
# #
# class dataFolder(data.Dataset):
#     #这边先加载训练数据的名字，因为数据太大了
#     def __init__(self,root,dataPerLenFile,):
#         dataNames=[]
#         for filename in sorted(glob.glob(root)):
#             #下行是用来判断生成的文件是否含有MPEG或者MVUB字符串，实际不用也可以
#             if is_ply_file(filename):
#                 dataNames.append(filename)
#         self.dataNames=dataNames
#         self.fileLen=len(self.dataNames)
#
#
#     #这里的代码没看懂，没看懂为什么只返回occupancy作为data
#     def __getitem__(self, index):
#         while (self.index + self.TreePoint > self.datalen):
#             #第
#             filename = self.dataNames[self.fileIndx]
#             # print(filename)
#             if self.dataBuffer:
#                 a = [self.dataBuffer[0][self.index:].copy()]
#             else:
#                 a = []
#             cell, mat = self.loader(filename)
#
#             #这里取的data单个样本信息，shape为[nodeNum,paretnK,number of elements(occupancy(1),level(1),octant(1),Pos(3),Color(3))]
#             for i in range(cell.shape[1]):
#                 data = np.transpose(mat[cell[0, i]])  # shape[ptNum,Kparent, Seq[1],Level[1],Octant[1],Pos[3] ] e.g 123456*7*6
#                 data[:, :, 0] = data[:, :, 0] - 1
#                 a.append(data[:, -parentNum:, :])  # only take levelNumK level feats
#
#             self.dataBuffer = []
#             self.dataBuffer.append(np.vstack(tuple(a)))
#
#             self.datalen = self.dataBuffer[0].shape[0]
#             self.fileIndx += 200  # shuffle step = 1, will load continuous mat
#             self.index = 0
#             if (self.fileIndx >= self.fileLen):
#                 self.fileIndx = index % self.fileLen
#         # print(index)
#         # try read
#         img = []
#         img.append(self.dataBuffer[0][self.index:self.index + self.TreePoint])
#
#         self.index += self.TreePoint
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img
#
#     def __len__(self):
#         return len(self.dataNames)
#
#
#     #计算所有文件中节点的平均数量
#     def CalcDataLenPerFile(self):
#         dataLenPerFile=0
#         for filename in self.dataNames:
#             cell,mat=plyLoader(filename)
#             dataLenPerFile+=mat[cell[0,0]].shape[2]
#         dataLenPerFile=dataLenPerFile/self.fileLen
#         print('dataLenPerFile:',dataLenPerFile)
#         self.dataLenPreFile=dataLenPerFile
#         return dataLenPerFile
#
#
# # #test is_ply_file功能
# # fileName1="MPEG_longdress_vox10_1052.mat"
# # print(any(extension in fileName1 for extension in PLY_EXTENSION))
#
#
# #test plyLoader
# filePath='D:\PythonFilesForPycharm\Octattention-mywork\octattention-final\Mywork\dataprepare\Data\TestTrain\MPEG_longdress_vox10_1052.mat'
# cell,mat=plyLoader(filePath)
# # print(cell.shape[1])
# # data=np.transpose(mat[cell[0,0]]) #shape=[nodeNum,Kparent,elementNums]--->(286012,4,9)
# # print(data.shape)
# # print(data[:,:,0])
# # print(data)
# # print(cell)
# # print(mat)
# # a=[]
# # for i in range(cell.shape[1]):
# #     data = np.transpose(mat[cell[0, i]])  # shape[ptNum,Kparent, Seq[1],Level[1],Octant[1],Pos[3] ] e.g 123456*7*6
# #     data[:, :, 0] = data[:, :, 0] - 1
# #     a.append(data[:, -parentNum:, :])  # only take levelNumK level feats



import numpy as np
import glob
import torch.utils.data as data
import glob
import h5py

IMG_EXTENSIONS = [
    'MPEG',
    'MVUB'
]
levelNumK=4


def is_image_file(filename):
    return any(extension in filename for extension in IMG_EXTENSIONS)


def default_loader(path):
    mat = h5py.File(path)
    # data = scio.loadmat(path)
    cell = mat['patchFile']
    return cell, mat


class DataFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, TreePoint, dataLenPerFile, transform=None, loader=default_loader):

        # dataLenPerFile is the number of all octnodes in one 'mat' file on average

        dataNames = []
        for filename in sorted(glob.glob(root)):
            if is_image_file(filename):
                dataNames.append('{}'.format(filename))
        self.root = root
        self.dataNames = sorted(dataNames)
        self.transform = transform
        self.loader = loader
        self.index = 0
        self.datalen = 0
        self.dataBuffer = []
        self.fileIndx = 0
        self.TreePoint = TreePoint
        self.fileLen = len(self.dataNames)
        assert self.fileLen > 0, 'no file found!'
        self.dataLenPerFile = dataLenPerFile  # you can replace 'dataLenPerFile' with the certain number in the 'calcdataLenPerFile'
        # self.dataLenPerFile = self.calcdataLenPerFile() # you can comment this line after you ran the 'calcdataLenPerFile'

    def calcdataLenPerFile(self):
        dataLenPerFile = 0
        for filename in self.dataNames:
            cell, mat = self.loader(filename)
            for i in range(cell.shape[1]):
                dataLenPerFile += mat[cell[0, i]].shape[2]
        dataLenPerFile = dataLenPerFile / self.fileLen
        print('dataLenPerFile:', dataLenPerFile, 'you just use this function for the first time')
        return dataLenPerFile

    def __getitem__(self, index):
        while (self.index + self.TreePoint > self.datalen):
            filename = self.dataNames[self.fileIndx]
            # print(filename)
            if self.dataBuffer:
                a = [self.dataBuffer[0][self.index:].copy()]
            else:
                a = []
            cell, mat = self.loader(filename)
            for i in range(cell.shape[1]):
                data = np.transpose(
                    mat[cell[0, i]])  # shape[ptNum,Kparent, Seq[1],Level[1],Octant[1],Pos[3] ] e.g 123456*7*6
                data[:, :, 0] = data[:, :, 0] - 1
                a.append(data[:, -levelNumK:, :])  # only take levelNumK level feats

            self.dataBuffer = []
            self.dataBuffer.append(np.vstack(tuple(a)))

            self.datalen = self.dataBuffer[0].shape[0]
            self.fileIndx += 200  # shuffle step = 1, will load continuous mat
            self.index = 0
            if (self.fileIndx >= self.fileLen):
                self.fileIndx = index % self.fileLen
        # print(index)
        # try read
        img = []
        img.append(self.dataBuffer[0][self.index:self.index + self.TreePoint])

        self.index += self.TreePoint

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return int(self.dataLenPerFile * self.fileLen / self.TreePoint)  # dataLen = octlen in total/TreePoint


# if __name__ == "__main__":
#
#     TreePoint = 4096 * 16  # the number of the continuous occupancy code in data, TreePoint*batch_size divisible by batchSize
#     batchSize = 32
#     train_loader = data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
#     print('total octrees(TreePoint*7): {}; total batches: {}'.format(len(train_set), len(train_loader)))
#
#     for batch, d in enumerate(train_loader):
#         data_source = d[0].reshape((batchSize, -1, 4, 6)).permute(1, 0, 2, 3)  # d[0] for geometry,d[1] for attribute
#         print(batch, data_source.shape)
#         # print(data_source[:,0,:,0])
#         # print(d[0][0],d[0].shape)
# # %%
#
if __name__ =="__main__":
    bptt = 2
    batchSize = 4
    TreePoint = bptt * batchSize
    trainDataRoot = 'D:\PythonFilesForPycharm\Octattention-mywork\octattention-final\Mywork\dataprepare\Data\TestTrain\*.mat'
    train_set = DataFolder(root=trainDataRoot, TreePoint=TreePoint, transform=None,
                           dataLenPerFile=13.6)
    # train_loader = data.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True, num_workers=4, drop_last=True)
    # for batch, d in enumerate(train_loader):
    #     data_source = d[0].reshape((batchSize, -1, 4, 6)).permute(1, 0, 2, 3)  # d[0] for geometry,d[1] for attribute
    #     print(batch, data_source.shape)
    #     print(data_source[:,0,:,0])
    #     print(d[0][0],d[0].shape)


    txtRoot = "D:\PythonFilesForPycharm\Octattention-mywork\octattention-final\Mywork\dataprepare\Data\\txt"
    for i, data in enumerate(train_set):
        txtFileRoot = txtRoot + '\\' + str(i) + '.txt'
        newData = np.array(data[0], dtype=str)
        with open(txtFileRoot, "w") as file:
            for d in newData:
                for line in d:
                    file.write(' '.join(line) + '\n')
                file.write('\n')
        #len(train_set)=dataLenPerFile * fileLen / TreePoint,
        # 这块我理解的是确保说每次取到Treepoint的bptt的数据都是相邻的点云来的
        # 含义就是不是将一个点的信息作为一个数据，而是将一个batch中的数据先综合取成(TreePoint，parentK，number of element)
        # 然后在Data，loader中按照batch_size取单个batch的时候就会让相邻的点靠在一起，取出来的batch就是(batch_size,bptt,parentK,number of element)
        if i == int(13.6 * 5 / TreePoint):
            break
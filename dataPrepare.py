'''
author:huangzelong@nuaa.edu.cn
time:2024.3.3
decriptions:Mywork
'''

import glob
import os
import numpy as np
import pandas as pd
import hdf5storage
from pointCloudTool import ptread
from GenOctree import genOctree
from GenKparentSeq import genKparentSeq

def makedFile(FilePath):
    fileList=sorted(glob.glob(FilePath))
    return fileList


def dataPrepare(fileName, color_format, saveMatDir,parentK, qs=1, ptNamePrefix='', offset='min', qlevel=None,
                rotation=False, normalize=False):
    if not os.path.exists(saveMatDir):
        os.makedirs(saveMatDir)

    # 准备数据的名字拼接--hzl
    ptName = ptNamePrefix+fileName.split('/')[-1][:-4]

    # 读入点云数据--hzl
    if (color_format == 'rgb'):
        p, color = ptread(fileName, color_format)
    else:
        p = ptread(fileName, color_format)

    # ptWithColor = np.hstack((p, color))
    # showPly(ptWithColor,"before")
    '''对颜色进行处理，这里重点是考虑最小单元内的颜色需要合并的问题，
    所有是不是需要先构建八叉树，在构建八叉树的时候，去处理'''
    # # print(ptName+"'s position:",p)
    # # print(ptName+"'s color:",color)
    # #
    refPt = p
    #
    #这里不影响颜色和坐标的对应关系
    if normalize is True:  # normalize pc to [-1,1]^3
        p = p - np.mean(p, axis=0)
        p = p / abs(p).max()
        refPt = p

    #这里也不影响颜色和坐标的对应关系
    if rotation:
        # [x,y,z]变成了[x,z,y]--hzl
        refPt = refPt[:, [0, 2, 1]]

        # [x,z,y]变成了[x,-z,y]--hzl
        refPt[:, 2] = - refPt[:, 2]

    #这里也不影响颜色和坐标的关系
    if offset == 'min':
        offset = np.min(refPt, 0)

    # 位置进行了一个平移--hzl
    points = refPt - offset

    #在规定八叉树层级的情况下，去计算步长
    if qlevel is not None:
        # 这里如果需要限制八叉树的深度，则需要说修改步长，这里为什么要减1？--hzl
        qs = (points.max() - points.min()) / (2 ** qlevel - 1)


    # 对位置坐标进行量化，这一步就已经完成了最小单元的节点位置归中操作，后续对同一位置的节点颜色平均即可完成最小单元内颜色的平均操作--hzl
    pt = np.round(points / qs)
    #
    # # 对定位到相同位置的点进行去重，对颜色取平均值--hzl
    # pt,idx = np.unique(pt,axis=0,return_index=True)
    #
    # # 数据类型强制转化--hzl
    pt = pt.astype(int)
    # # pointCloud.write_ply_data('pori.ply',np.hstack((pt,c)),attributeName=['reflectance'],attriType=['uint16'])
    #
    # '''
    # 这里需要处理一下位置坐标,其中只有unique会影响原来的顺序,同时会去除一些点,
    # 这里我做的方法是在坐标处理完之后,将原本颜色拼接,之后对位置相同的点颜色取平均,
    # 同时这里用np.unique要慢于pd.unique,所以这里也将np.unique替换为pd.unique
    # '''
    if (color_format == 'geometry'):
        pt = np.unique(pt, return_index=True)
    else:
        ptWithColor = np.hstack((pt, color))
        ptWithColor = pd.DataFrame(ptWithColor, columns=['x', 'y', 'z', 'r', 'g', 'b'])
        ptWithColor = ptWithColor.groupby(['x', 'y', 'z'], as_index=False).mean()
        # print(ptWithColor)
        ptWithColor = np.array(ptWithColor)
        p = ptWithColor[:, 0:3]
        color=ptWithColor[:,3:6]

    #点云可视化
    # ptWithColor=ptWithColor[0:30000,:]
    # showPly(ptWithColor,"after")
    # print(p,color)
    # 生成八叉树，返回层次遍历的八叉树的节点的Occupance，还有整个树的结构，以及八叉树的深度,其中八叉树节点携带相应颜色--hzl
    Codes, octree, Lmax=genOctree(p,color)
    visualize_octree_and_point_cloud(octree, p, color)
    octree[0].level=Lmax

    # for test--hzl
    print(Codes)
    # print(octree)
    print(Lmax)
    for L in range(Lmax):
        print("第"+str(L)+"层:")
        for index,node in enumerate(octree[L].nodes):
            # print("index:",index)
            # print("node.nodeid:",node.node_id)
            # print("node.occupancy:",node.occupancy)
            print("node.pos:",node.pos*qs+offset)
            print("node.color:",node.color)
            # print("node.child:",node.child_points)
            # print("node.octant:",node.octant)
            # print("node.parent_id:",node.parent_id)


    # Info={"Codes":Codes,"octree":octree,"Lmax":Lmax}
    # hdf5storage.savemat(os.path.join(outDirPath, ptName + '.mat'), Info)

    # code, Octree, QLevel = GenOctreeByPy(pt)
    # if (color_format == 'rgb'):
    #     colorSeq = ColourOctree(Octree, ptWithColor, 4)
    DataStruct = genKparentSeq(octree, parentK)
    #
    ptcloud = {'Location': refPt}
    Info = {'qs': qs, 'offset': offset, 'Lmax': Lmax, 'name': ptName,
            'levelSID': np.array([Octreelevel.nodes[-1].node_id for Octreelevel in octree])}
    patchFile = {'patchFile':(np.concatenate((np.expand_dims(DataStruct['occupancySeq'],2),DataStruct['Level'],DataStruct['Pos'],DataStruct['Color']),2), ptcloud, Info)}
    #在patchFile中,数据被合并成(num,K,9),这里9是1(occupancy)+2(LevelAndOctant)+3(Pos)+3(Color)
    # print(DataStruct['occupancySeq'].shape)
    # print(DataStruct['Level'].shape)
    # print(DataStruct['Pos'].shape)
    # print(DataStruct['Color'].shape)

    hdf5storage.savemat(os.path.join(saveMatDir,ptName+'.mat'), patchFile, format='7.3', oned_as='row', store_python_metadata=True)
    DQpt = (pt*qs+offset)
    return os.path.join(saveMatDir,ptName+'.mat'),DQpt,refPt

    #for test
    # Info={'position':p,'color':color}
    # # print(Info)
    # print(ptName+'.mat')
    # hdf5storage.savemat(os.path.join(outDirPath,ptName+'.mat'),Info)
    # return

class CPrintl():
    def __init__(self, logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName) != '' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))

    def __call__(self, *args):
        # print(*args)
        print(*args, file=open(self.log_file, 'a'))



inDirPath='/root/autodl-tmp/Data/Test/longdress_vox10_1052.ply'
outDirPath='/root/autodl-tmp/Data/TestTrain'
ptNamePrefix = 'MPEG_'
parentNum=4
printl = CPrintl('./Log/makedFileObj.log')
makedList = makedFile(outDirPath + '/*.mat')
fileList = sorted(glob.glob(inDirPath))
if __name__=="__main__":
    for n,file in enumerate(fileList):
        outMatName=outDirPath+'\\'+ptNamePrefix+file.split('\\')[-1][:-4]+'.mat'
        # print(fileName)
        # print(outMatName)
        if outMatName in makedList:
            print(outMatName+" maked!")
        else:
            dataPrepare(file,color_format="rgb",saveMatDir=outDirPath,parentK=parentNum,ptNamePrefix=ptNamePrefix,offset=0,rotation=False)
            print(outMatName)
            printl(outMatName)

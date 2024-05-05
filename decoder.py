'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
'''
#%%
from tqdm import tqdm
from GenOctree import DeOctree, dec2bin
from Tool import write_ply_data
from dataSet import default_loader as matloader
from collections import deque
import time
from Writepcerror import pcerror
from netWorkTool import *
from encoderTool import generate_square_subsequent_mask
from encoder import model,list_orifile
import numpyAc
import numpy as np
batch_size = 1 
bpttRepeatTime = 1
#%%
'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''
def decodeOct(binfile,oct_data_seq,model,bptt):
    model.eval()    #模型打开验证模式
    with torch.no_grad():   #不用构建计算图，开始推理，不用进行自动求导
        elapsed = time.time()   #开始计时

        KfatherNode = [[255,0,0]]*levelNumK     #K层祖先节点的信息，这里默认设置为[255,0,0]*4-->[occupancy,octant,level]
        nodeQ = deque()     #节点队列
        oct_seq = []    
        src_mask = generate_square_subsequent_mask(bptt).to(device)     #一个窗口中对于各个节点的掩码

        input = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)    #初始为全0，shape=[1024，1,4,3]
        padinginbptt = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device) #在编码阶段有对前后进行一个padding的操作，长度为bptt
        bpttMovSize = bptt//bpttRepeatTime
        # input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
        # all of [oct,level,octant] default is zero

        output = model(input,src_mask,[])   #这里输入的是全0，可以对比一下是不是一样的数据，shape=[1024,1,255],需要修改一下为[1024,4,256]
        #根节点的概率分布
        freqsinit = torch.softmax(output[-1],1).squeeze().cpu().detach().numpy()    #把概率最后一个拿出来，->[1,255]做一个softmax，然后squeeze一下
        
        oct_len = len(oct_data_seq)

        dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)   #算术解码器,传入参数为oct_len(5466),概率分布维度为255，读入文件为binfile

        root =  decodeNode(freqsinit,dec)   #传入参数为概率分布，以及C++解码后端框架
        nodeId = 0
        
        KfatherNode = KfatherNode[3:]+[[root,1,1]] + [[root,1,1]] # for padding for first row # ( the parent of root node is root itself)
        
        nodeQ.append(KfatherNode) 
        oct_seq.append(root) #decode the root  
        
        with tqdm(total=  oct_len+10) as pbar:
            while True:
                father = nodeQ.popleft()
                childOcu = dec2bin(father[-1][0])
                childOcu.reverse()
                faterLevel = father[-1][1] 
                for i in range(8):
                    if(childOcu[i]):
                        faterFeat = [[father+[[root,faterLevel+1,i+1]]]] # Fill in the information of the node currently decoded [xi-1, xi level, xi octant]
                        faterFeatTensor = torch.Tensor(faterFeat).long().to(device)
                        faterFeatTensor[:,:,:,0] -= 1

                        # shift bptt window
                        offsetInbpttt = (nodeId)%(bpttMovSize) # the offset of current node in the bppt window
                        if offsetInbpttt==0: # a new bptt window
                            input = torch.vstack((input[bpttMovSize:],faterFeatTensor,padinginbptt[0:bpttMovSize-1]))
                        else:
                            input[bptt-bpttMovSize+offsetInbpttt] = faterFeatTensor

                        output = model(input,src_mask,[])
                        
                        Pro = torch.softmax(output[offsetInbpttt+bptt-bpttMovSize],1).squeeze().cpu().detach().numpy()

                        root =  decodeNode(Pro,dec)
                        nodeId += 1
                        pbar.update(1)
                        KfatherNode = father[1:]+[[root,faterLevel+1,i+1]]
                        nodeQ.append(KfatherNode)
                        if(root==256 or nodeId==oct_len):
                            assert len(oct_data_seq) == nodeId # for check oct num
                            Code = oct_seq
                            return Code,time.time() - elapsed
                        oct_seq.append(root)
                    assert oct_data_seq[nodeId] == root # for check

def decodeNode(pro,dec):
    root = dec.decode(np.expand_dims(pro,0))    #传入概率分布，shpae=[1,255]
    return root+1


if __name__=="__main__":

    for oriFile in list_orifile: # from encoder.py
        ptName = os.path.basename(oriFile)[:-4]
        matName = '/root/tf-logs/SourceCode/OctAttention-obj/Data/LootTest/'+ptName+'.mat'
        binfile = expName+'/data/'+ptName+'.bin'
        cell,mat =matloader(matName)
        # print(cell.shape[1])
        # data=
        print(mat[cell[0,0]])
        # for i in range(cell.shape[1]):
        #     data = np.transpose(mat[cell[0, i]])  # shape[ptNum,Kparent, Seq[1],Level[1],Octant[1],Pos[3] ] e.g 123456*7*6
        #     data[:, :, 0] = data[:, :, 0] - 1
        #     a.append(data[:, -levelNumK:, :])  # only take levelNumK level feats
        # Read Sideinfo
        oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-1:,0]# for check 真实ocupancy值
        
        color=np.transpose(mat[cell[0,3]]).astype(int)[:,-1:,0]
        
        p = np.transpose(mat[cell[1,0]]['Location']) # ori point cloud  #原始点云中的位置坐标
        offset = np.transpose(mat[cell[2,0]]['offset']) #dataprepare中的预处理阶段得到的offset
        qs = mat[cell[2,0]]['qs'][0]    #dataprepare阶段确定的量化参数

        Code,elapsed = decodeOct(binfile,oct_data_seq,model,bptt)
        print('decode succee,time:', elapsed)
        print('oct len:',len(Code))

        # DeOctree
        ptrec = DeOctree(Code)
        # Dequantization
        DQpt = (ptrec*qs+offset)
        write_ply_data(expName + "/temp/test/rec.ply", DQpt)
        pcerror(p, DQpt, None, '-r 1', None).wait()
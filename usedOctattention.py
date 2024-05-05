'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-20 08:06:11
LastEditTime: 2021-09-20 23:53:24
LastEditors: fcy
Description: the training file
             see networkTool.py to set up the parameters
             will generate training log file loss.log and checkpoint in folder 'expName'
FilePath: /compression/dataSet.py
All rights reserved.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import datetime
from netWorkTool import *
from torch.utils.tensorboard import SummaryWriter
from usedNetModel import TransformerLayer, TransformerModule

##########################

ntokens = 256  # the size of vocabulary
ninp = 4 * (128 + 4 + 6+ 64*3)  # embedding dimension=parentK*(128+4+6+382)=4*512,这里将颜色的embedding设置到382

nhid = 300  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # the number of heads in the multiheadattention models
dropout = 0  # the dropout value
batchSize = 32



#自定义的loss，单个点的loss应当为位置的交叉熵损失加上颜色的交叉熵损失，同时这里颜色与位置应当不能为相同的权重
'''
在位置相差较大的时候，位置损失较为明显， 
'''
#output.shape=[点的数量，4,256]
weight=[0.7,0.1,0.1,0.1]
def my_cross_entropy(outputs, targets):
    ls=nn.LogSoftmax(dim=2)
    OLS=ls(outputs)#outputLogSoftmax
    # #拆分通道
    # OLSforOccupancy=outputLogSoftmax[:,0,:]
    # OLSforColor1=outputLogSoftmax[:,1,:]
    # OLSforColor2=outputLogSoftmax[:,2,:]
    # OLSforColor3=outputLogSoftmax[:,3,:]
    
    pointNum=targets.size(0)
    chanels=targets.size(1)
    loss=0  #累计loss值,每个点的loss是由两部分来组成的
    for b in range(pointNum):
        index=targets[b]
        loss+=OLS[b,:,index].dot(weight)
    loss=loss*(-1)
    return loss

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 128)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL + 1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        #forColor
        self.encoder3 = nn.Embedding(256,64)
        self.encoder4=nn.Embedding(256,64)
        self.encoder5=nn.Embedding(256,64)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(ninp, ninp)
        self.Occupancy = nn.Linear(ninp, ntoken)
        self.Color1=nn.Linear(ninp,ntoken)
        self.Color2=nn.Linear(ninp,ntoken)
        self.Color3=nn.Linear(ninp,ntoken)
        
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data)
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data = nn.init.xavier_normal_(self.decoder0.weight.data)
        self.Occupancy.bias.data.zero_()
        self.Occupancy.weight.data = nn.init.xavier_normal_(self.Occupancy.weight.data)
        
        self.Color1.bias.data.zero_()
        self.Color1.weight.data = nn.init.xavier_normal_(self.Color1.weight.data)
        self.Color2.bias.data.zero_()
        self.Color2.weight.data = nn.init.xavier_normal_(self.Color2.weight.data)
        self.Color3.bias.data.zero_()
        self.Color3.weight.data = nn.init.xavier_normal_(self.Color3.weight.data)


    def forward(self, src, src_mask, dataFeat):
        bptt = src.shape[0]
        batch = src.shape[1]

        oct = src[:, :, :, 0]  # oct[bptt,batchsize,FeatDim(levels)] [0~254]
        level = src[:, :, :, 1]  # [0~12] 0 for padding data
        octant = src[:, :, :, 2]  # [0~8] 0 for padding data
        color1= src[:,:,:,6]    #for color
        color2=src[:,:,:,7]
        color3=src[:,:,:,8]
        
        # assert oct.min()>=0 and oct.max()<255
        # assert level.min()>=0 and level.max()<=12
        # assert octant.min()>=0 and octant.max()<=8
        # assert color.min()>=0 and color.max()<=255

        level -= torch.clip(level[:, :, -1:] - 10, 0, None)  # the max level in traning dataset is 10
        torch.clip_(level, 0, MAX_OCTREE_LEVEL)
        aOct = self.encoder(oct.long())  # a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())
        # aColor=self.encoder3(color.long())
        aColor1=self.encoder3(color1.long())
        aColor2=self.encoder4(color2.long())
        aColor3=self.encoder5(color3.long())

        a = torch.cat((aOct, aLevel, aOctant,aColor1,aColor2,aColor3), 3)

        a = a.reshape((bptt, batch, -1))

        # src = self.ancestor_attention(a)
        src = a.reshape((bptt, a.shape[1], self.ninp)) * math.sqrt(self.ninp)

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output=self.act(self.decoder0(output))
        output0 = self.Occupancy(output)
        output1 = self.Color1(output)
        output2 = self.Color2(output)
        output3 = self.Color3(output)
        output = torch.cat((output0.unsqueeze(2), output1.unsqueeze(2), output2.unsqueeze(2), output3.unsqueeze(2)), dim=2)
        return output


######################################################################
# ``PositionalEncoding`` module
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

#source.sape=[batch_size,bptt,parentK,number of element]
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len].clone()
    target = source[i + 1:i + 1 + seq_len, :, -1, 0].reshape((-1))# [bptt*batch_size,4]-->[bptt*batchsize*4]
    my_target=source[i + 1:i + 1 + seq_len, :, -1, [0,6,7,8]].reshape((-1,4))
    '''这里的操作是生成当前的训练数据文件，对于下一个点，我需要去预测他的占用码
    则我需要获取到下一节点的K层祖先节点的信息，则存放在source[i+1:i+1+seq_len：0：-1，：]
    之中，所以训练数据首先要将原本的节点的对应祖先节点信息替换成需要预测的节点的祖先信息，
    即为data[:,:0:-1,:]=source[i+1：i+1+seq_len,:,0：-1，：]
    此时data的最后一行没有变，还是当前节点的信息，但是对于预测下一个节点而言，下一个节点的Octant
    和level是已知的，所以也将下一个节点的octant和level放进训练数据中（？？？这里没懂为什么octant和level是已知的）
    
    经过作者指点说，在解码端是广度优先的，解码到对应的点自然就知道了
    
    '''
    data[:, :, 0:-1, :] = source[i + 1:i + seq_len + 1, :, 0:-1,:]  # this moves the feat(octant,level) of current node to lastrow,
    data[:, :, -1, 1:3] = source[i + 1:i + seq_len + 1, :, -1, 1:3]  # which will be used as known feat
    return data[:, :, -levelNumK:, :], (target).long(), [],(my_target).long()


######################################################################
# Run the model
# -------------
#
torch.cuda.empty_cache()
model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
if __name__ == "__main__":
    import dataSet
    import torch.utils.data as data
    import time
    import os

    epochs = 8  # The number of epochs
    best_model = None
    batch_size = 128
    TreePoint = bptt * 16

    train_set = dataSet.DataFolder(root=trainDataRoot, TreePoint=TreePoint, transform=None,
                                   dataLenPerFile=312310.61666666664)  # you should run 'dataLenPerFile' in dataset.py to get this num (17456051.4)
    # train_set.calcdataLenPerFile()
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)  # will load TreePoint*batch_size at one time

    # loger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName + '/loss.log')
    writer = SummaryWriter('./log/' + expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    # model_structure(model,printl)
    printl(expComment + ' Pid: ' + str(os.getpid()))
    log_interval = int(batch_size * TreePoint / batchSize / bptt)
    # log_interval=1

    # learning
    criterion = [nn.CrossEntropyLoss() for i in range(4)]
    # criterion=nn.CrossEntropyLoss()
    lr = 1e-3  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    # saveDic = reload(100030,checkpointPath)
    if saveDic:
        scheduler.last_epoch = saveDic['epoch'] - 1
        idloss = saveDic['idloss']
        best_val_loss = saveDic['best_val_loss']
        model.load_state_dict(saveDic['encoder'])


    def train(epoch):
        global idloss, best_val_loss
        model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1, 7))

        for Batch, d in enumerate(
                train_loader):  # there are two 'BATCH', 'Batch' includes batch_size*TreePoint/batchSize/bptt 'batch'es.
            batch = 0

            train_data = d[0].reshape((batchSize, -1, 4, 9)).to(device).permute(1, 0, 2,
                                                                                3)  # shape [bptt*batch_size,batch_size,4,9]
            src_mask = model.generate_square_subsequent_mask(bptt).to(device)#这个mask是和窗口大小匹配的
            for index, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets, dataFeat,my_target = get_batch(train_data, i)  # data:[bptt,batch_size,4_9],target->[bptt*batch_size*4],targets->[4,-1]
                optimizer.zero_grad()
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask, dataFeat)  # output: [bptt,batch size,4,256]
                
                #这里需要重新设计一下损失函数，最好将颜色与位置的损失相关联起来
                #使得说位置偏差越大，整体的损失就越大，位置相差不多时，则更体现颜色的损失
                '''
                这里需要对交叉熵损失函数进行魔改，实现带权重的交叉熵损失函数
                参考如下链接:https://zhuanlan.zhihu.com/p/369683910
                
                '''
                # weight = torch.Tensor([7, 1, 1, 1]).long()
                output=output.view(-1,4,ntokens)
                loss=0
                for i in range(4):
                    loss+=criterion[i](output[:,i,:],my_target[:,i])
                loss=loss/(math.log(2)*4)
                # loss = criterion(output.view(-1,ntokens), targets) / math.log(2)
                # loss=my_cross_entropy(output.view(-1,4,ntokens), my_target)/math.log(2)
                print(loss)
                # print('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | loss {:5.2f}'.format(epoch, Batch, batch,loss))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                batch = batch + 1

                if batch % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time

                    total_loss_list = " - "
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                           'lr {:02.2f} | ms/batch {:5.2f} | '
                           'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                        epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                                             elapsed * 1000 / log_interval,
                        cur_loss, total_loss_list, math.exp(cur_loss)))
                    total_loss = 0

                    start_time = time.time()

                    writer.add_scalar('train_loss', cur_loss, idloss)
                    idloss += 1

            if Batch % 10 == 0:
                save(epoch * 100000 + Batch, saveDict={'encoder': model.state_dict(), 'idloss': idloss, 'epoch': epoch,
                                                       'best_val_loss': best_val_loss}, modelDir=checkpointPath)


    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f'---------------epoch={epoch}--------------')
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)

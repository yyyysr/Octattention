import h5py
from netWorkTool import levelNumK,expName,MAX_OCTREE_LEVEL,bptt,device
import numpy as np
import os
import torch
import tqdm
import time
import numpyAc
from writePly import write_ply_file

#这块我有点疑问，主要是不知道颜色取当前的还是下个点的，目前跟着occupancy操作
def batchify(oct_seq,bptt,oct_len):
    oct_seq[:-1,0:-1,:] = oct_seq[1:,0:-1,:]
    oct_seq[:-1,-1,1:3] = oct_seq[1:,-1,1:3]
    oct_seq[:,:,0] = oct_seq[:,:,0] - 1
    pad_len = bptt#int(np.ceil(len(oct_seq)/bptt)*bptt - len(oct_seq))
    oct_seq = torch.Tensor(np.r_[np.zeros((bptt,*oct_seq.shape[1:])),oct_seq,np.zeros((pad_len,*oct_seq.shape[1:]))])
    dataID = torch.LongTensor(np.r_[np.ones((bptt))*-1,np.arange(oct_len),np.ones((pad_len))*-1])
    return dataID.unsqueeze(1),oct_seq.unsqueeze(1)


def encodeNode(pro,octvalue):
    # assert octvalue<=255 and octvalue>=1
    pre = np.argmax(pro)+1
    return -np.log2(pro[octvalue-1]+1e-07),int(octvalue==pre)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def compress(node_data_seq, outputfile,model, actualcode=True, print=print, showRelut=False):
    model.eval()
    levelID = node_data_seq[:, -1, 1].copy()
    node_data_seq = node_data_seq.copy()

    if levelID.max() > MAX_OCTREE_LEVEL:
        print('**warning!!**,to clip the level>{:d}!'.format(MAX_OCTREE_LEVEL))

    oct_seq = node_data_seq[:, -1:, 0].astype(int)      #需要编码的原始occupancy
    color0_seq = node_data_seq[:, -1:, 6].astype(int)   #需要编码的原始color0
    color1_seq = node_data_seq[:, -1:, 7].astype(int)   #需要编码的原始color1
    color2_seq = node_data_seq[:, -1:, 8].astype(int)   #需要编码的原始color2
    oct_len = len(oct_seq)  
    

    batch_size = 1  # 1 for safety encoder

    assert (batch_size * bptt < oct_len)

    # %%
    dataID, padingdata = batchify(node_data_seq, bptt, oct_len)
    MAX_GPU_MEM_It = 2 ** 13  # you can change this according to the GPU memory size (2**12 for 24G)
    MAX_GPU_MEM = min(bptt * MAX_GPU_MEM_It, dataID.max()) + 2  # bptt <= MAX_GPU_MEM -1 < min(MAX_GPU,dataID)
    pro = torch.zeros((4,MAX_GPU_MEM, 256)).to(device)
    padingLength = padingdata.shape[0]
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    padingdata = padingdata
    elapsed = 0
    offset = 0
    proBit = [[] for _ in range(4)]
    if not showRelut:
        trange = range
    else:
        trange = tqdm.trange
    with torch.no_grad():
        for n, i in enumerate(trange(0, padingLength - bptt, bptt)):
            input = padingdata[i:i + bptt].long().to(
                device)  # input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
            nodeID = dataID[i + 1:i + bptt + 1].squeeze(0) - offset
            nodeID[nodeID < 0] = -1
            start_time = time.time()
            output = model(input, src_mask, []) #[1024,1,4,256]
            output0=output[:,:,0,:]
            output1=output[:,:,1,:]
            output2=output[:,:,2,:]
            output3=output[:,:,3,:]
            elapsed = elapsed + time.time() - start_time
            output0 = output0.reshape(-1, 256)
            output1 = output1.reshape(-1, 256)
            output2 = output2.reshape(-1, 256)
            output3 = output3.reshape(-1, 256)
            nodeID = nodeID.reshape(-1)
            p0 = torch.softmax(output0, 1)
            p1 = torch.softmax(output1, 1)
            p2 = torch.softmax(output2, 1)
            p3 = torch.softmax(output3, 1)
            pro[0,nodeID, :] = p0
            pro[1,nodeID, :] = p1
            pro[2,nodeID, :] = p2
            pro[3,nodeID, :] = p3
            if ((n % MAX_GPU_MEM_It == 0 and n > 0) or n == padingLength // bptt - 1):
                proBit[0].append(pro[0,:nodeID.max() + 1].detach().cpu().numpy())
                proBit[1].append(pro[1,:nodeID.max() + 1].detach().cpu().numpy())
                proBit[2].append(pro[2,:nodeID.max() + 1].detach().cpu().numpy())
                proBit[3].append(pro[3,:nodeID.max() + 1].detach().cpu().numpy())
                offset = offset + nodeID.max() + 1

    del pro, input, src_mask
    torch.cuda.empty_cache()
    proBit = np.vstack(proBit)
    # %%

    bit0 = []
    bit1 = []
    bit2 = []
    bit3 = []
    acc0 = []
    acc1 = []
    acc2 = []
    acc3 = []
    templevel = 1
    binszList0 = []
    binszList1 = []
    binszList2 = []
    binszList3 = []
    octNumList=[]
    if True:
        # Estimate the bitrate at each level
        for i in range(oct_len):
            octvalue = int(oct_seq[i, -1])
            color0value=int(color0_seq[i,-1])
            color1value=int(color1_seq[i,-1])
            color2value=int(color2_seq[i,-1])
            bit0, acc0 = encodeNode(proBit[0], octvalue)
            bit1, acc1 = encodeNode(proBit[1], color0value)
            bit2, acc2 = encodeNode(proBit[2], color1value)
            bit3, acc3 = encodeNode(proBit[3], color2value)
            if templevel != levelID[i]:
                templevel = levelID[i]
                binszList0.append(bit0)
                binszList1.append(bit1)
                binszList2.append(bit2)
                binszList3.append(bit3)
                octNumList.append(i + 1)
        binszList0.append(bit0)
        binszList1.append(bit1)
        binszList2.append(bit2)
        binszList3.append(bit3)
        octNumList.append(i + 1)
        binszOccupancy = bit0  # estimated bin size
        binszColor0 = bit1  # estimated bin size
        binszColor1 = bit2  # estimated bin size
        binszColor2 = bit3  # estimated bin size

        if actualcode:
            codec = numpyAc.arithmeticCoding()
            if not os.path.exists(os.path.dirname(outputfile)):
                os.makedirs(os.path.dirname(outputfile))
            _,binsz0 = codec.encode(proBit[0,:oct_len, :], oct_seq.astype(np.int16).squeeze(-1) - 1, outputfile0)
            _,binsz1 = codec.encode(proBit[1,:oct_len,:], color0_seq.astype(np.int16).squeeze(-1)-1,outputfile1)
            _,binsz2 = codec.encode(proBit[2,:oct_len,:], color1_seq.astype(np.int16).squeeze(-1)-1,outputfile2)
            _,binsz3 = codec.encode(proBit[3,:oct_len,:], color2_seq.astype(np.int16).squeeze(-1)-1,outputfile3)
        binsz=binsz0+binsz1+binsz2+binsz3
        binszList=torch.cat((binszList0.unsqueeze(0), binszList1.unsqueeze(0), binszList2.unsqueeze(0), binszList3.unsqueeze(0)), dim=0)
        if len(binszList) <= 7:
            return binsz, oct_len, elapsed, np.array(binszList), np.array(octNumList)
        return binsz, oct_len, elapsed, np.array(binszList[:,7:]), np.array(octNumList[7:])
        # %%




def matloader(path):
    mat = h5py.File(path)
    # data = scio.loadmat(path)
    cell = mat['patchFile']
    return cell,mat


def main(fileName, model, actualcode=True, showRelut=True, printl=print):
    #建树文件之后生成的mat文件
    matDataPath = fileName
    octDataPath = matDataPath
    cell, mat = matloader(matDataPath)
    FeatDim = levelNumK

    node_data_seq = np.transpose(mat[cell[0, 0]]).astype(int)[:, -FeatDim:, 0:9]   #所有节点的[nodeNum,4,9]
    
    node_data=node_data_seq[:,-1:,3:]
    plyfile="/root/autodl-tmp/Data/RQPly/"+fileName.split('/')[-1][:-4]+'.ply'
    write_ply_file(node_data,plyfile)

    p = np.transpose(mat[cell[1, 0]]['Location'])   #真实坐标
    ptNum = p.shape[0]      #真实点的数量
    ptName = os.path.basename(matDataPath)  
    PrefixName=['Occupancy','Color0','Color1','Color2']     #四个bin文件的前缀
    outputfile=[]
    for i,prefix in enumerate(PrefixName):
        outputfile.append(expName+"/data/" + ptName[:-4]+prefix + ".bin")
    binsz, oct_len, elapsed, binszList, octNumList = compress(node_data_seq, outputfile, model, actualcode, printl,
                                                              showRelut)
    if showRelut:
        printl("ptName: ", ptName)
        printl("time(s):", elapsed)
        printl("ori file", octDataPath)
        printl("ptNum:", ptNum)
        printl("binsize(b):", binsz)
        printl("bpip:", binsz / ptNum)

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        printl("pre sz(b) from Q8:", (binszList))
        printl("pre bit per oct from Q8:", (binszList / octNumList))
        printl('octNum', octNumList)
        printl("bit per oct:", binsz / oct_len)
        printl("oct len", oct_len)

    return binsz / oct_len
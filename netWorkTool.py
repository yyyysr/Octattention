# bptt = 1024 # Context window length
# batchSize = 1
# dataLenPerFile=294395.5 #from dataset.DataFolder.cala…… function

import os
import torch

MAX_OCTREE_LEVEL=12
bptt=1024
levelNumK=4
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expName = '/root/autodl-tmp/Data/Exp/Obj'
DataRoot = '/root/autodl-tmp/Data/Train'
checkpointPath = expName+'/checkpoint'
trainDataRoot = DataRoot+"/*.mat" # DON'T FORGET RUN ImageFolder.calcdataLenPerFile() FIRST
expComment = 'OctAttention, trained on MPEG 8i,MVUB 1~10 level. 2023/3. All rights reserved.'


def save(index, saveDict, modelDir='checkpoint', pthType='epoch'):
    if os.path.dirname(modelDir) != '' and not os.path.exists(os.path.dirname(modelDir)):
        os.makedirs(os.path.dirname(modelDir))
    torch.save(saveDict, modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, index))


class CPrintl():
    def __init__(self, logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName) != '' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))

    def __call__(self, *args):
        print(*args)
        print(*args, file=open(self.log_file, 'a'))
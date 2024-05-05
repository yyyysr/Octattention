from usedOctattention import model
from encoderTool import main
from netWorkTool import expName,device,levelNumK
import torch
import datetime,os
from dataPrepare import dataPrepare
from Writepcerror import pcerror


def reload(checkpoint,modelDir='checkpoint',pthType='epoch',print=print,multiGPU=False):
    try:
        if checkpoint is not None:
            saveDict = torch.load(modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint),map_location=device)
            pth = modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint)
        if checkpoint is None:
            saveDict = torch.load(modelDir,map_location=device)
            pth = modelDir
        saveDict['path'] = pth
        # print('load: ',pth)
        if multiGPU:
            from collections import OrderedDict
            state_dict = OrderedDict()
            new_state_dict = OrderedDict()
            for k, v in saveDict['encoder'].items():
                name = k[7:]  # remove `module.`
                state_dict[name] = v
            saveDict['encoder'] = state_dict
        return saveDict
    except Exception as e:
        print('**warning**',e,' start from initial model')
        # saveDict['path'] = e
    return None


class CPrintl():
    def __init__(self,logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName)!='' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))
    def __call__(self, *args):
        print(*args)
        print(*args, file=open(self.log_file, 'a'))

model = model.to(device)
saveDic = reload(None, '/root/autodl-tmp/Data/Exp/Obj/checkpoint/encoder_epoch_00800040.pth')
model.load_state_dict(saveDic['encoder'])

###########Objct##############
list_orifile = ['/root/autodl-tmp/Data/EncodeTest/loot_vox10_1001.ply']
if __name__ == "__main__":
    printl = CPrintl(expName + '/encoderPLY.txt')
    printl('_' * 50, 'OctAttention V0.4', '_' * 50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    # printl('load checkpoint', saveDic['path'])
    for oriFile in list_orifile:
        printl(oriFile)
        if (os.path.getsize(oriFile) > 300 * (1024 ** 2)):  # 300M
            printl('too large!')
            continue
        ptName = os.path.splitext(os.path.basename(oriFile))[0]
        for qs in [1]:
            ptNamePrefix = ptName
            
            #def dataPrepare(fileName, color_format, saveMatDir,parentK, qs=1, ptNamePrefix='', offset='min', qlevel=None,rotation=False, normalize=False):
            matFile, DQpt, refPt = dataPrepare(oriFile, color_format='rgb',parentK=levelNumK,saveMatDir='/root/autodl-tmp/Data/EncodeTestMat', qs=qs, ptNamePrefix='',
                                               rotation=False)
            # please set `rotation=True` in the `Mywork` function when processing MVUB data
            main(matFile, model, actualcode=True, printl=printl)  # actualcode=False: bin file will not be generated
            print('_' * 50, 'pc_error', '_' * 50)
            pcerror(refPt, DQpt, None, '-r 1023', None).wait()

import dataSet as dataSet
import torch.utils.data as data


DataRoot = '/root/autodl-tmp/Data/TestTrain'
trainDataRoot = DataRoot+"/*.mat" # DON'T FORGET RUN ImageFolder.calcdataLenPerFile() FIRST
bptt=4
TreePoint=bptt * 2
batch_size=2


if __name__=="__main__":
    
    train_set = dataSet.DataFolder(root=trainDataRoot, TreePoint=TreePoint, transform=None,
                                    dataLenPerFile=13.25)  # you should run 'dataLenPerFile' in dataset.py to get this num (17456051.4)
    # train_set.calcdataLenPerFile()
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,
                                    drop_last=True)  # will load TreePoint*batch_size at one time
    for n,data in enumerate(train_loader):
        print(data[0].shape)
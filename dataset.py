# generate dataset
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import torch
import copy

def get_dataloader(args):
    test_dataset=None
    if args.dataset=="Coloured dSprites":
        pass
    elif args.dataset=="3dchairs":
        data=np.load("data/3dchairs.npy")
        print(data.shape)
        print(np.sum(np.isnan(data)))
        assert np.sum(np.isnan(data))==0
        data=torch.tensor(data.astype(np.float32)).float()/255
        dataset=TensorDataset(data)
        pass
    elif args.dataset=="celeba":
        data=np.load("data/celeba.npy")
        print(data.shape)
        print(np.sum(np.isnan(data)))
        assert np.sum(np.isnan(data))==0
        data=torch.tensor(data.astype(np.float32)).float()/255
        dataset=TensorDataset(data)
        pass
    elif args.dataset=="dsprites":
        data=np.load("data/dsprites.npz", encoding='bytes')
        np.random.shuffle(data['imgs'])
        np.random.shuffle(data['imgs'])
        index=np.array( list(range(data['imgs'].shape[0])) )
        np.random.shuffle(index)
        train_index=index[0:-1000]
        test_index=index[-1000:-1]


        traindata = torch.from_numpy(copy.deepcopy(data['imgs'][train_index])).unsqueeze(1).float()
        testdata = torch.from_numpy(copy.deepcopy(data['imgs'][test_index])).unsqueeze(1).float()
        

        dataset=TensorDataset(traindata)
        test_dataset=TensorDataset(testdata)
        pass
    else:
        raise ValueError("data set not found! ")
    dataloader=DataLoader(dataset,shuffle=True,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    if test_dataset is not None:
        testdataloader=DataLoader(test_dataset,shuffle=True,batch_size=200,num_workers=args.workers,pin_memory=True)
    else:
        testdataloader=None
    return dataloader,testdataloader

if __name__=="__main__":
        data=np.load("data/3dchairs.npy")
        print(data.shape)
        print(data.dtype)
        print(data[0,:,:,:].max())
        print(data[0,:,:,:].min())
        print(np.sum(np.isnan(data)))
        assert np.sum(np.isnan(data))==0
        data=data.astype(np.float32)
        print(data[0,:,:,:].max())
        print(data[0,:,:,:].min())

        data=torch.tensor(data).float()
        print(data[0,:,:,:].max())
        print(data[0,:,:,:].min())
        # print(data.float())
        dataset=TensorDataset(data)

def load_model(path):
    model=torch.load("results/"+path)
    return model

def processing(path):
    im=Image.open(path).resize((64,64),PIL.Image.BILINEAR).convert("RGB")
    im=np.array(im).transpose(2,0,1)
    return im

# def load_test_data(dataset):
#     image_list=[]
#     if dataset=="celeba":
#         path = "data/data/CelebA/img_align_celeba/test_data/"
#         for img in os.listdir(path):
#             image_list.append(processing(path+img)/255)
#     elif dataset=="chairs":
#         path = "data/data/3DChairs/images/test_data/"
#         for img in os.listdir(path):
#             image_list.append(processing(path+img)/255)
#     return image_list
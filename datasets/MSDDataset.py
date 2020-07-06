from torch.utils.data import Dataset
import os, os.path
import skimage, skimage.transform
from skimage.io import imread, imsave
from PIL import Image
import skimage.filters
import json
import medpy, medpy.io
import numpy as np
import collections
import torchvision.transforms
import torchvision.transforms.functional as TF
import h5py, ntpath
import torch
import time

import sys
sys.path.append('../utils')
import general

def extract_samples(data, image_path, label_path):
    image_data, _ = medpy.io.load(image_path)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    labels = seg_data.sum((1,2)) > 1

    print (collections.Counter(labels))

    for i in range(image_data.shape[0]):
        data.append((image_data[i],seg_data[i],labels[i]))

def extract_samples2(data, labels, image_path, label_path):
    new_size = 224
    image_data, _ = medpy.io.load(image_path)
    print('image_data :', image_data.shape)
    image_data = image_data.transpose(2,0,1)
    seg_data, _ = medpy.io.load(label_path)
    seg_data = seg_data.transpose(2,0,1)
    these_labels = seg_data.sum((1,2)) > 1

    print(collections.Counter(these_labels))

    for i in range(image_data.shape[0]):
        
        # slice의 크기를 224x224로 맞춰준다
        img_resize = skimage.transform.resize(image_data[i], (new_size, new_size))
        seg_resize = skimage.transform.resize(seg_data[i], (new_size, new_size))
        seg_resize = (seg_resize > 0) * 1.0
        
        data.append([img_resize, seg_resize])
        labels.append(these_labels[i])


def compute_hdf5(dataroot, files, hdf5_name):
    if not os.path.exists(os.path.split(hdf5_name)[0]):
        os.makedirs(os.path.split(hdf5_name)[0])
    with h5py.File(hdf5_name,"w") as hf:
        files = sorted(files, key=lambda k: k["image"])
        for i, p in enumerate(files):
            print(p["image"], p["label"])
            name = ntpath.basename(p["image"])

            grp = hf.create_group(name)
            grp.attrs['name'] = name
            grp.attrs['author'] = "jpc"

            samples = []
            labels = []

            extract_samples2(samples, labels, dataroot +'/'+ p["image"], dataroot +'/'+ p["label"])

            grp_slices = grp.create_group("slices")
            for idx, zlice in enumerate(samples):
                print(".", end=" ")
                grp_slices.create_dataset(str(idx),data=zlice, compression='gzip')
            print(".")
            grp.create_dataset("labels",data=labels)

def scale(image, maxval=1024):
    """Assumes that maxvalue and minvalue are the same."""
    image += maxval # minimum value is now 0
    image /= maxval*2

    return(image)


def randomCrop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask


def ToTensor(image):
    """Convert ndarrays in sample to Tensors."""

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image.copy())


#https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606
def transform(image, mask, is_train, new_size, mask_size, aug): # new_size는 학습할 때, mask_size는 IoU계산할 떄 사용
        if aug:
            if is_train:
                # Resize
                image = skimage.transform.resize(image, (new_size+20, new_size+20))
                mask = skimage.transform.resize(mask, (new_size+20, new_size+20))

                # Random crop
                image, mask = randomCrop(image, mask, new_size, new_size)

                # Random horizontal flipping
                if np.random.random() > 0.5:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)

                # Random vertical flipping
                if np.random.random() > 0.5:
                    image = np.flipud(image)
                    mask = np.flipud(mask)
            else:
                # Resize
                image = skimage.transform.resize(image, (new_size, new_size))
                mask = skimage.transform.resize(mask, (new_size, new_size))
                
                
        # IoU를 계산할 때와 학습할 떄 mask의 크기가 다르기 때문에 이렇게 한번 더 처리해줌
        if mask_size is not None:
            # Transform to tensor
#             mask_old = mask
            mask = skimage.transform.resize(mask, (mask_size, mask_size))
    
        image = ToTensor(image)
        mask = ToTensor(mask)
#         mask_old = ToTensor(mask_old)
                                            
        # Scale image to (maximum) ~[0 1]
        #image = scale(image)
        return image, mask #, mask_old




def wrap_setattr(attr, value):
    def foo(func):
        setattr(func, attr, value)
        return func
    return foo

def setdatasetname(value):
    return wrap_setattr('_DG_NAME', value)

cached_msd_ref = {}





@setdatasetname("MSDDataset")
class MSDDataset(Dataset):
    def __init__(self, mode, args): # mode, dataroot, blur=0, seed=0, nsamples=32, maxmasks=1, transform=None, new_size=100, mask_all=False, 

#         assert 0 <= maxmasks <= 1
        self.mode = mode
        self.dataroot = args.data_path
        self.maxtrain = args.train_ratio # train data 중에 이 만큼만 사용한다는 뜻
        self.maxmasks = args.mask_ratio if self.mode == "train" else 1.0 # 전체 mask중에 이 만큼만 mask 사용한다는 뜻
        self.depth = args.input_depth
        self.mask_size = args.mask_data_size
        self.aug = args.data_aug
        self.nsamples = None
        self.seed = args.trial
        blur = 0
        
        
        assert self.depth%2 == 1, 'depth should be an odd number'

        # 처음 MSDdataset을 불러왔을 때는 아래 나오는 os.path.basename(self.dataroot)에 ori_data폴더를 만들고 거기에
        # imagesTr, imagesTs, labelsTr, dataset.json를 넣어야한다. 이것들은 .tar파일을 풀면 생김. 이렇게만 하고 돌리면 
        # dataset_in_use.hdf5는 알아서 생김
        self.filename = os.path.join(self.dataroot, "dataset_in_use.hdf5")
        if not os.path.isfile(self.filename):
            print("Computing hdf5 file of the data")
            ori_data_path = os.path.join(os.path.split(self.dataroot)[0], 'ori_data')
            dataset = json.load(open(os.path.join(ori_data_path, "dataset.json")))
            files = dataset['training']
            compute_hdf5(ori_data_path, files, self.filename)
        
        # image, mask의 size를 원래 저장되어 있는 사이즈로 설정함(나중에 바꿀 일있으면 다른 값으로 입력)
        with h5py.File(self.filename, "r") as f:
            self.new_size = f[list(f.keys())[0]]['slices']['0'].shape[-1]
        # store cached reference so we can load the valid and test faster
#         if not self.dataroot in cached_msd_ref:
#             cached_msd_ref[self.dataroot] = h5py.File(filename,"r")
#         self.dataset = cached_msd_ref[self.dataroot]
        
        
        with h5py.File(self.filename, "r") as f:

            self._all_files = sorted(list(f.keys()))

            all_labels = np.concatenate([f[i]["labels"] for i in self._all_files])
            print ("Full dataset contains: " + str(collections.Counter(all_labels)))


            np.random.seed(self.seed)
            
            print("mode=" + self.mode)
            
            self.get_files_to_use() # output은 self.files와 self.nsamples에 저장됨
            
            print("Loading {} files:".format(len(self.files)) + str(self.files))
            #self.samples = np.concatenate([self.dataset[i]["slices"] for i in self.files])

            # 위에서 train이 선택됐으면 train에 해당하는 patient들의 
            # 모든 슬라이드와 mask를 self.sample, self.labels에 모은다.
            # list self.mask_files에 있는 환자들만 mask를 사용한다
            self.samples = []
            self.mask_files = np.random.choice(self.files, round(len(self.files) * self.maxmasks), replace=False)
            for file in self.files:
                total_slice_num = len(f[file]["slices"])

                # 데이터 하나의 depth가 n이면 한 환자당 데이터는 n-1개 줄어든다
                for sli in range(total_slice_num-(self.depth-1)):
                    use_mask_or_not = 1 if file in self.mask_files else 0
                    self.samples.append((file, sli, use_mask_or_not))

            print('Selected {} files for masks : {}'.format(len(self.mask_files), self.mask_files))
                    
            # 데이터 하나의 depth가 n이면 가운데를 기준으로 label이 결정되므로 양쪽 끝 label (n-1)//2개가 사라져야 한다.
            # 참고 : 나중에 label을 이용해서 실제 사용할 image와 label의 idx를 self.idx에 저장할 것임.
            if self.depth != 1:
                self.labels = np.concatenate([f[i]["labels"][(self.depth-1)//2:-(self.depth-1)//2] for i in self.files])
            else:
                self.labels = np.concatenate([f[i]["labels"] for i in self.files])

            assert len(self.samples) == len(self.labels), 'Num of image and mask are different'

            
            print('################ self.labels :', self.labels.shape)
            
            # pos와 neg의 개수를 맞추기 위해 둘 중 개수가 적은 것의 x2 만큼 사용
            if args.small_dataset and self.mode != 'test': 
                self.nsamples = 32 * 3 # 작은 데이터셋으로 테스트로 돌려보고 싶을 때
            elif self.mode != 'test':
                self.nsamples = min(np.sum(self.labels == 1), np.sum(self.labels == 0)) * 2

                
                
            #self.transform = transform
            self.blur = blur

            print ("Loaded images contain:" + str(collections.Counter(self.labels)))

            
            
            self.idx = np.arange(self.labels.shape[0])

            # randomly choose based on nsamples
            # pos, neg label 개수를 맞춰주려면 많은 쪽에서 데이터를 샘플해서 개수를 줄여야 함
            n_per_class = self.nsamples//2
            np.random.seed(self.seed)
            class0 = np.where(self.labels == 0)[0]
            class1 = np.where(self.labels == 1)[0]
            class0 = np.random.choice(class0, n_per_class, replace=False)
            class1 = np.random.choice(class1, n_per_class, replace=False)
            self.idx = np.append(class1, class0) # 두 일차원 벡터를 길게 붙임

            #these should be in order
            #self.samples = self.samples[self.idx]
            self.labels = self.labels[self.idx]

            
            
            # masks_selector is 1 for samples that should have a mask, else zero.
            self.masks_selector = np.ones(len(self.idx))
            if self.mode == "train":
                if self.maxmasks < 1:
                    n_masks_to_rm = int(round(n_per_class * (1-self.maxmasks)))
                    idx_masks_class1_to_rm = np.random.choice(
                        np.arange(n_per_class), n_masks_to_rm, replace=False)
                    idx_masks_class0_to_rm = np.random.choice(
                        np.arange(n_per_class, n_per_class*2), n_masks_to_rm,
                        replace=False)

                    self.masks_selector[idx_masks_class0_to_rm] = 0
                    self.masks_selector[idx_masks_class1_to_rm] = 0

                
                
            print ("This dataloader contains: {}\n".format(
                str(collections.Counter(self.labels))))

            # NB: transform does nothing, only exists for compatibility.

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        
#         start = time.time()
        
        with h5py.File(self.filename, "r") as f:
    
            key = self.samples[self.idx[index]] # self.samples는 ('liver_0.nii.gz', '123')이 담겨있음. 환자와 그 환자의 몇번째 slice인지
            image, seg = [], []
            for num in range(self.depth):
    #             print(str(key[1]+num))
                image_tmp, seg_tmp = f[key[0]]["slices"][str(key[1]+num)] # self.dataset는 hdf5 읽을 때 쓰는 f와 같음
                image.append(image_tmp)
                seg.append(seg_tmp)
            image = np.stack(image, axis=2)
            seg = np.stack(seg, axis=2).mean(2, keepdims=True)
            label = self.labels[index]
            use_mask = key[2]

    #         print(image.shape, seg.shape)

            # Make the segmentation the entire image if it isn't in masks_selector.
    #         if not self.masks_selector[index]:
    #             seg = np.zeros(seg.shape)

            # Make the segmentation the entire image if it is the negative class.
    #         if int(label) == 0:
    #             seg = np.ones(seg.shape)

            # If there is a segmentation, blur it a bit.
    #         if (self.blur > 0) and (seg.max() != 0):
    #             seg = skimage.filters.gaussian(seg, self.blur)
    #             seg = seg / seg.max()

    #         seg = (seg > 0) * 1.

    #         print(time.time() - start, '1')
    #         start = time.time()
            
            # data가 synthetic일 때는 flipping을 하면 안되므로 is_train을 False로 한다
            if self.mode == "train" and 'synthetic' not in self.dataroot:
                image, seg = transform(image, seg, True, self.new_size, self.mask_size, self.aug)
            else:
                image, seg = transform(image, seg, False, self.new_size, self.mask_size, self.aug)

    #         print(time.time() - start, '2')
        return (image, seg), int(label), use_mask
    
    def get_files_to_use(self,):
        # data가 synthetic일 때는 train과 val로 쓸 데이터가 이미 지정되어 있기 때문에 랜덤으로 나누지 않는다.
        if 'synthetic' in self.dataroot:
            if self.mode == "train":                
                self.files = [file for file in self._all_files if 'train' in file]
                if self.maxtrain < 1:
                    self.files = np.random.choice(self.files, round(len(self.files) * self.maxtrain), replace=False)
            elif self.mode == "valid":
                self.files = [file for file in self._all_files if 'val' in file]
            elif self.mode == "test":
                self.nsamples = 100 # test가 아닌 경우엔 뒤에서 정의
                self.files = [file for file in self._all_files if 'val' in file]
            else:
                raise Exception("Unknown mode")
        else:
            np.random.seed(self.seed)
            np.random.shuffle(self._all_files)
            file_ratio = int(len(self._all_files)*0.8)

            if self.mode == "train":                
                self.files = self._all_files[:file_ratio]
                if self.maxtrain < 1:
                    self.files = np.random.choice(self.files, round(file_ratio * self.maxtrain), replace=False)
            elif self.mode == "valid":
                self.files = self._all_files[file_ratio:]
            elif self.mode == "test":
                self.nsamples = 100 # test가 아닌 경우엔 뒤에서 정의
                self.files = self._all_files[file_ratio:]
            else:
                raise Exception("Unknown mode")
                
        

# @setdatasetname("LungMSDDataset")
# class LungMSDDataset(MSDDataset):
#     def __init__(self, **kwargs):
#         super().__init__(dataroot='/media/volume1/juna/LRP_project/data_msd/lung/Task06_Lung/', **kwargs) 
#         # /network/data1/MSD/MSD/Task06_Lung/

# @setdatasetname("ColonMSDDataset")
# class ColonMSDDataset(MSDDataset):
#     def __init__(self, **kwargs):
#         super().__init__(dataroot='/media/volume1/juna/LRP_project/data_msd/colon/Task10_Colon/', **kwargs) 
#         # /network/data1/MSD/MSD/Task10_Colon/

# @setdatasetname("LiverMSDDataset")
# class LiverMSDDataset(MSDDataset):
#     def __init__(self, **kwargs):
#         super().__init__(dataroot='/media/volume1/juna/LRP_project/data_msd/liver/Task03_Liver/', **kwargs) 
#         # /network/data1/MSD/MSD/Task03_Liver/

# @setdatasetname("PancreasMSDDataset")
# class PancreasMSDDataset(MSDDataset):
#     def __init__(self, **kwargs):
#         super().__init__(dataroot='/media/volume1/juna/LRP_project/data_msd/pancreas/Task07_Pancreas/', **kwargs) 
#         # /network/data1/MSD/MSD/Task07_Pancreas/

# @setdatasetname("ProstateMSDDataset")
# class ProstateMSDDataset(MSDDataset):
#     def __init__(self, **kwargs):
#         super().__init__(dataroot='/media/volume1/juna/LRP_project/data_msd/prostate/Task05_Prostate/', **kwargs) 
#         # /network/data1/MSD/MSD/Task05_Prostate/

# @setdatasetname("HeartMSDDataset")
# class HeartMSDDataset(MSDDataset):
#     def __init__(self, **kwargs):
#         super().__init__(dataroot='/media/volume1/juna/LRP_project/data_msd/heart/Task02_Heart/', **kwargs)
        

# setdatasetname(or setmodelname)이 선언되면 결국 return되는 건 foo함수의 인자로 들어간 func 클래스다. 
# decorator는 이 클래스의 attribution중에 '_DG_NAME'(or 'MODEL_NAME')안에 value를 넣어서 다시 내보낸다.
# 결과적으로 setdatasetname(or setmodelname)는 다음 줄에 있는 class의 attribution '_DG_NAME'(or 'MODEL_NAME')에
# (없다면 만들어서) decorator의 인자인 "PancreasMSDDataset" 이런 것들을 넣어 주는 역할을 한다.


import sys
sys.path.append('../../utils')
import general

import time
import h5py
import skimage.transform
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print ("    %s: %s".format(key, val))

with h5py.File('./2.size_224/dataset_in_use_.hdf5', 'w') as f:
    pass

new_size = 224

data = None
with h5py.File('1.size_512/dataset_in_use.hdf5', 'r') as f:
    keys = list(f.keys())
    keys.sort()
    
    # patient 하나를 집어든다
    for patient_num in tqdm(range(131)):
        total_img_num = len(f['liver_{}.nii.gz'.format(patient_num)]['slices'].keys())

        label = f['liver_{}.nii.gz'.format(patient_num)]['labels'][()]
        if total_img_num != label.shape[0]:
            raise Exception('total img num and total label num is different / img:{}, label:{}'.format(total_img_num, label.shape[0]))
        
        True_idx = np.where(label == True)[0]
        
        # 간과 간 사이에 간이 아닌 slice가 있는 것들이 있는지 체크
        if True_idx[-1] - True_idx[0]+1 != True_idx.shape[0]:
            print('Different!!!!!'.format(patient_num))
            print(label)
            print(True_idx)
            
#             # 간과 간 사이에 간이 아닌 slice가 있는 것들은 slice가 적은 쪽을 False로 바꿔줌
#             if patient_num == 94:
#                 print(f['liver_{}.nii.gz'.format(patient_num)]['labels'][395:400])
#                 f['liver_{}.nii.gz'.format(patient_num)]['labels'][395] = False
#                 print(f['liver_{}.nii.gz'.format(patient_num)]['labels'][395:400])
#             elif patient_num == 102:
#                 print(f['liver_{}.nii.gz'.format(patient_num)]['labels'][322:330])
#                 f['liver_{}.nii.gz'.format(patient_num)]['labels'][322:325] = False
#                 print(f['liver_{}.nii.gz'.format(patient_num)]['labels'][322:330])
        else:
#             print('Same')
            pass
            
        # 그 patient의 하나의 slice를 집어든다
        count = 0
        new_label_list = []
        for num in range(total_img_num):
#             print('label : {}'.format(label[num]))
#             img = f['liver_{}.nii.gz'.format(patient_num)]['slices'][str(num)][()]
#             img = np.transpose(img, (1, 2, 0))
#             general.imshow_depth(img)

#             img_reshape = img.reshape(-1)
#             plt.hist(img_reshape)
            
            # 그 slice가 간에 해당하면
            if label[num]:
                img = f['liver_{}.nii.gz'.format(patient_num)]['slices'][str(num)][0]
                mask = f['liver_{}.nii.gz'.format(patient_num)]['slices'][str(num)][1]
                assert mask.sum() != 0, 'This mask has only zero values'
                
                # mask를 tumor만 남김
                new_mask = (mask > 1) * 1.0
                
                # slice에 tumor가 존재하면
                if new_mask.sum() > 0:
                    # new_label에 tumor가 존재한다고 저장
                    new_label_list.append(True)
                else:
                    new_label_list.append(False)
                
                # mask가 너무 작은 건 label을 False로 바꿔준다.
                boundary = new_mask.size / 2000
                if new_label_list[-1] and (new_mask.sum() < boundary):
                    new_label_list[-1] = False
                    new_mask = np.zeros_like(new_mask)
                
                img = skimage.transform.resize(img, (new_size, new_size))
                new_mask = skimage.transform.resize(new_mask, (new_size, new_size))
                
                pair = np.stack([img, new_mask], axis=0)
                
                general.save_hdf5('./2.size_224/dataset_in_use_.hdf5', 
                                  'liver_{}.nii.gz/slices/{}'.format(patient_num, count), pair, 
                                  no_maxshape=True, 
                                  compression=True)
                count += 1
        
        new_label = np.array(new_label_list)
        general.save_hdf5('./2.size_224/dataset_in_use_.hdf5', 'liver_{}.nii.gz/labels'.format(patient_num), new_label, no_maxshape=True)

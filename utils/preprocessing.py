import os
import numpy as np
import h5py
import nibabel as nib
import matplotlib.pyplot as plt

def nib_to_liver_range(filename):
    # input  : nib filename
    # output : array of tumor range ex) np.array([135, 230])
    seg = nib.load(filename)
    seg_data = np.array(seg.get_fdata())
    print('{} : {}'.format(os.path.basename(filename), seg_data.shape))
    
    last_slice_sum = 0
    slice_sum = 0
    liver_range = []
    for z_num in range(seg_data.shape[2]):
        last_slice_sum = slice_sum
        slice_sum = np.sum(seg_data[:,:,z_num])
        
        if slice_sum > 0 and last_slice_sum == 0:
            liver_range.append(z_num)
        elif slice_sum == 0 and last_slice_sum > 0:
            liver_range.append(z_num)
            
    print('liver_range :', liver_range)
    if len(liver_range) != 2:
        print('More or less than two!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        new_liver_range = []
        new_liver_range.append(liver_range[len(liver_range)-2])
        new_liver_range.append(liver_range[len(liver_range)-1])
        print('new_liver_range :', new_liver_range)
        
        liver_range = new_liver_range
    
    return np.array(liver_range)




# input : nib filename
def get_liver_n_tumor_location_list(filename, all_liver_range, liver_num):
    only_liver_point = []
    tumor_location_point = []
    
    seg = nib.load(filename)
    seg_data = np.array(seg.get_fdata())
    print('{} : {}'.format(os.path.basename(filename), seg_data.shape))
    
    liver_start = all_liver_range[liver_num, 0]
    liver_end = all_liver_range[liver_num, 1]
    print('liver_start liver_end :', liver_start, liver_end)
    
    tumor_or_liver_count = 0
    for z_num in range(seg_data.shape[2]):
        if 2 in seg_data[:,:,z_num]:
            if tumor_or_liver_count == 0:
                print('tumor start z_num :', z_num)
                if z_num < liver_start:
                    print('z_num < liver_start !$#%$^@%&#^*$&*%#&$@^#%@#!$^@%&#^#%@^$@%&#^*^$&')
                tumor_or_liver_count += 1
            location_point = (z_num - liver_start) / (liver_end - liver_start + 1)
            print('tumor location_point :', location_point, z_num)
            tumor_location_point.append(location_point)
            
        elif 2 not in seg_data[:,:,z_num] and 1 in seg_data[:,:,z_num]:
            location_point = (z_num - liver_start) / (liver_end - liver_start + 1)
            print('liver location_point :', location_point, z_num)
            only_liver_point.append(location_point)
            
            
    if tumor_or_liver_count == 0:
        print('This is healthy liver!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    print()
    
    return tumor_location_point, only_liver_point





def get_vol_seg_tag_from_nib(vol_filename, seg_filename, liver_num, liver_num_start, all_liver_range):
    vol = nib.load(vol_filename)
    vol_data = np.array(vol.get_fdata())
    print('volume-{:03d}.nii : {}'.format(liver_num, vol_data.shape))
    
    seg = nib.load(seg_filename)
    seg_data = np.array(seg.get_fdata())
    print('segmentation-{:03d}.nii : {}'.format(liver_num, seg_data.shape))
    
    liver_start = all_liver_range[liver_num-liver_num_start, 0]
    liver_end = all_liver_range[liver_num-liver_num_start, 1]
    print('liver_start liver_end :', liver_start, liver_end)
    
    tag = []
    location = []
    for z_num in range(liver_start, liver_end):
        if 2 in seg_data[:,:,z_num]:
            print('tumor z :', z_num)
            tag.append(1) # 1의 뜻은 tumor
            location.append((z_num - liver_start) / (liver_end - liver_start)) # liver_end은 liver의 마지막 slice의 넘버에서 하나 큰 수 
            
        elif 2 not in seg_data[:,:,z_num] and 1 in seg_data[:,:,z_num]:
            print('liver z :', z_num)
            tag.append(0) # 0의 뜻은 liver
            location.append((z_num - liver_start) / (liver_end - liver_start)) # liver_end은 liver의 마지막 slice의 넘버에서 하나 큰 수 
    
    if (liver_end - liver_start) != len(tag):
        raise Exception('Error : (liver_end - liver_start),  len(tag) are not the same')
    vol_data = vol_data[:,:,liver_start:liver_end]
    seg_data = seg_data[:,:,liver_start:liver_end]
    tag = np.array(tag)
    location = np.array(location)
    
    return vol_data, seg_data, tag, location



# output : numpy array
def get_rem_other_vol_seg_tag(filename):
    vol_rem_other = None
    seg_rem_other = None
    tag = None
    
    print(filename)
    with h5py.File(filename, 'r') as f:
        vol_rem_other = f['vol_rem_other'][()]
        seg_rem_other = f['seg_rem_other'][()]
        tag = f['tag'][()]
        location = f['location'][()]
        
    return vol_rem_other, seg_rem_other, tag, location


def get_window_from_np(vol_rem_other, seg_rem_other, tag, total_location, window_num):
    all_indices = np.arange(vol_rem_other.shape[2])
    all_indices = list(all_indices[:-(window_num-1)])
    print('vol_rem_other.shape[2] :', vol_rem_other.shape[2])
    print('len(all_indices) :', len(all_indices))
    print('all_indices :', all_indices)
    
    data = []
    mask = []
    label = []
    location = []
    for index in all_indices:
        vol_rem_other_win = vol_rem_other[:,:,index:index+window_num]
        seg_rem_other_win = seg_rem_other[:,:,index:index+window_num]
        label_win = tag[index+int((window_num-1)/2)] # 0 : liver, 1 : tumor
        location_win = total_location[index+int((window_num-1)/2)]
    
        data.append(vol_rem_other_win)
        mask.append(seg_rem_other_win)
        label.append(label_win)
        location.append(location_win)
        
    print('data :', len(data))
    print('mask :', len(mask))
    print('label :', len(label))
    print('location :', len(location))
    
    data = np.stack(data, axis=0)
    mask = np.stack(mask, axis=0)
    label = np.array(label)
    location = np.array(location)
    
    return data, mask, label, location






















































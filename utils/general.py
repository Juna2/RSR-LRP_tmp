import os
import re
import cv2
import h5py
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect



def extract_digit(my_str):
    return int(''.join(list(filter(str.isdigit, my_str))))

def extract_str(my_str):
    return ''.join([char for char in my_str if not char.isdigit()])

def save_hdf5(path, key, data, no_maxshape=False, compression=False):
    with h5py.File(path, 'a') as f:
        if key not in f: 
            if no_maxshape:
                f.create_dataset(key, data=data, compression=compression) # 압축하고 싶으면 compression='gzip'
            else:
                maxshape = tuple(None for i in range(len(data.shape)))
                f.create_dataset(key, data=data, maxshape=maxshape)
        else:
            f[key].resize((f[key].shape[0] + data.shape[0]), axis=0)
            f[key][-data.shape[0]:] = data

            
            
def imshow_depth(img, start_depth=0, showing_img_num=None, figsize=(20, 20)):
    depth = img.shape[2]
    if showing_img_num == None:
        fig, ax = plt.subplots(nrows=1, ncols=depth, figsize=figsize)
        for depth_num in range(depth):
            ax[depth_num].imshow(img[:,:,start_depth+depth_num], cmap='gray')
            ax[depth_num].axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=showing_img_num, figsize=(20, 20))
        for depth_num in range(showing_img_num):
            ax[depth_num].imshow(img[:,:,start_depth+depth_num], cmap='gray')
            ax[depth_num].axis('off')
        plt.show()
        
        
def imshow_depth_seaborn(img, start_depth=0, showing_img_num=None, cbar=True, figsize=(20, 20)):
    depth = img.shape[2]
    if showing_img_num == None:
        fig, ax = plt.subplots(nrows=1, ncols=depth, figsize=figsize)
        for depth_num in range(depth):
            seaborn.heatmap(img[:,:,start_depth+depth_num], cmap='RdBu_r', center=0, ax=ax[depth_num], square=True, cbar_kws={'shrink': .2})
        plt.axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=showing_img_num, figsize=(20, 20))
        for depth_num in range(showing_img_num):
            seaborn.heatmap(img[:,:,start_depth+depth_num], cmap='RdBu_r', center=0, ax=ax[depth_num], square=True, cbar=cbar, cbar_kws={'shrink': .2})
        plt.axis('off')
        plt.show()
        
        

           
def hist(array, start=None, end=None, unit=None, figsize=(10, 5)):
    if start == None:
        start = np.min(array)
    if end == None:
        end = np.max(array)
    if unit == None:
        unit = (end - start) / 10
    
    print('start : {}, end : {}, unit: {}'.format(start, end, unit))
    
    x = [start+unit*i for i in range(1000) if start+unit*i < end]
    x += [x[-1] + unit]
    y = [0 for i in range(len(x))]
    print('x :', x)
    print('y :', y)
    for i in range(array.shape[0]):
#         print(i)
        quo = int((array[i]-start) // unit)
        if 0 <= quo and quo <= len(y):
#             print(quo, array[i])
            y[quo] += 1
        
    #     print(y)
    # print()
    # print(x)
    # print(y)
    # print(sum(y))
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(y)), y, tick_label=[round(i, 1) for i in x], align='center')

    # plt.xlim( -1, len(y))
    # plt.ylim( 0, 400)
    plt.show()
            

def resize(img, shrink_size):
    # cv2.resize는 shape length가 2인 matrix만 resize할 수 있어서 한장한장 resize한 다음에 다 붙일거임
    other_dim = img.shape[:-2]
    if len(img.shape) > 2:
        img = img.reshape(np.prod(img.shape[:-2]), *img.shape[-2:])
    
    img_resized = []
    for num in range(img.shape[0]):
        one_resized_img = cv2.resize(img[num], shrink_size, interpolation = cv2.INTER_AREA)
        img_resized.append(np.expand_dims(one_resized_img, axis=0))

    img_resized = np.concatenate(img_resized, axis=0)
    img_resized = img_resized.reshape(*other_dim, *img_resized.shape[-2:])
    
    return img_resized
        
        
def make_result_dir(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
        
        
        
        
        
        
        
        
        
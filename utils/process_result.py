import os
import cv2
import h5py
import time
import torch
import psutil
import shutil
import seaborn
import general
import fnmatch
import collections
import numpy as np
import pandas as pd
import help_execute
import skimage.transform

from visdom import Visdom
import matplotlib.pyplot as plt
from scipy.special import softmax
from subprocess import Popen, PIPE
from matplotlib.figure import figaspect

def reset_previous_process(result_path):
    all_done_hypo_dir = [os.path.join(result_path, hypo_dir) for hypo_dir in os.listdir(os.path.join(result_path)) if '201' in hypo_dir or '202' in hypo_dir]
    all_done_hypo_dir.sort()
#     for hyp_dir in all_done_hypo_dir:
#         print(hyp_dir)
#     print('Start removing all result_total.hdf5')
    for hyp_dir in all_done_hypo_dir:
    #         print(os.path.join(hyp_dir, 'result_total.hdf5'))
        if 'result_total.hdf5' in os.listdir(hyp_dir):
            os.remove(os.path.join(hyp_dir, 'result_total.hdf5'))
#             print('{} has been removed.'.format(os.path.join(hyp_dir, 'result_total.hdf5')))
#     print('done')

#     print()
#     print('Start removing "_done" from all hyper_directories')
    for hyp_dir in all_done_hypo_dir:
        if 'done' in hyp_dir:
#             print(hyp_dir, hyp_dir[:-5])
            os.rename(hyp_dir, hyp_dir[:-5])
#     print('done')
    
    
def Is_this_directory_ready(result_path, directory):
    trial = None
    data_path = None
    with open(os.path.join(result_path, directory, 'setting.txt'), 'r') as f:
        for line in f:
            if 'trial' == line[:line.find(':')-1]:
                trial = int(line[line.find(':')+2:])
            elif 'data_path' == line[:line.find(':')-1]:
                data_path = line[line.find(':')+2:-1] # :-2 is to eliminate '\n'
    if trial == None:
        raise Exception('There is no trial in {} directory'.format(directory))
    elif data_path == None:
        raise Exception('There is no data_path in {} directory'.format(directory))
    
    folders = [file for file in os.listdir(os.path.join(data_path)) if 'dataset_in_use' in file]
    folders.sort()
    file = folders[-1]
    total_folder_num = int(0)+1
    
    files_in_directory = [file for file in os.listdir(os.path.join(result_path, directory)) if file[:3] == 'res']
    files_in_directory.sort()
    if len(files_in_directory) == 0:
        print('There is an empty directory : {}'.format(directory))
        return False
    elif len(files_in_directory) > total_folder_num:
        return False
    elif len(files_in_directory) < total_folder_num:
        print('Training is not done yet : {}'.format(directory))
        return False
    elif len(files_in_directory) == total_folder_num:
        with h5py.File(os.path.join(result_path, directory, files_in_directory[-1]), 'r') as f:
            if 'train_loss{}'.format(trial-1) in list(f.keys()):
                return True
            else:
                print('Training is not done yet : {}'.format(directory))
                return False

def total_trial_num(result_path, directory):
    trial = None
    with open(os.path.join(result_path, directory, 'setting.txt'), 'r') as f:
        for line in f:
            if 'trial' == line[:line.find(':')-1]:
                trial = int(line[line.find(':')+2:])
                return trial
    if trial == None:
        raise Exception('There is no trial in {} directory'.format(directory))
        
def total_folder_num(result_path, directory):
    data_path = None
    with open(os.path.join(result_path, directory, 'setting.txt'), 'r') as f:
        for line in f:
            if 'data_path' == line[:line.find(':')-1]:
                data_path = line[line.find(':')+2:-1] # :-2 is to eliminate '\n'
    if data_path == None:
        raise Exception('There is no data_path in {} directory'.format(directory))
    
    folders = [file for file in os.listdir(os.path.join(data_path)) if 'dataset_in_use' in file]
    folders.sort()
    file = folders[-1]
    total_folder_num = int(0)+1
    return total_folder_num

def get_multiple_data_sum(f, dataset_key_list):
    sum_of_datasets = None
    for index, key in enumerate(dataset_key_list):
        if index == 0:
            sum_of_datasets = f[key][()]
        else:
            sum_of_datasets += f[key][()]
    return sum_of_datasets

def make_dict_with_key_list(key_list, f):
    my_dict = collections.OrderedDict()
    for key in key_list:
        my_dict[key] = np.zeros_like(f['{}0'.format(key)]) # my_dict[key] = np.zeros_like(f[list(f.keys())[0]])
    return my_dict
        
def mean_results(filename, deno):
    my_dict = None
    with h5py.File(filename, 'r') as f:
        # train_loss0, val_loss0, ... 에서 숫자 빼고 str만 남긴 후 중복 걸러내고 key_set에 넣음
        key_set = set([general.extract_str(key) for key in list(f.keys())])
        
        # my_dict는 위에서 만든 key_set을 key로 가지는 dictionary
        my_dict = make_dict_with_key_list(key_set, f)
        
        # key val_accuracy에 val_accuracy0, val_accuracy1, val_accuracy2, ... 등을 모두 누적시켜서 더함.
        for my_key in list(my_dict.keys()):
            for key in list(f.keys()):
                if my_key in key:
                    my_dict[my_key] += f[key][()]
        
        # 더한 것들을 평균낸 후 내보냄
        for my_key in list(my_dict.keys()):
            my_dict[my_key] = my_dict[my_key] / deno
    return my_dict
                    
        
        
def get_mean_of_result(result_path):
    all_directories = [directory for directory in os.listdir(os.path.join(result_path)) 
                       if directory[:2] == '20' and Is_this_directory_ready(result_path, directory)]

    for directory in all_directories:
        files_in_directory = [file for file in os.listdir(os.path.join(result_path, directory)) if file[:3] == 'res']
        trial_num = total_trial_num(result_path, directory)
        folder_num = total_folder_num(result_path, directory)
        
        
        for file in files_in_directory: # cross validation할 때의 fold들이 files_in_directory. ex) result00, result01, ...
            # mean_results는 trial 또는 fold에 대한 평균을 계산해서 결과를 dictionary로 output을 만들어냄
            trial_mean_dict = mean_results(os.path.join(result_path, directory, file), trial_num)
#             print('trial_mean_dict :', trial_mean_dict)
            
            result_total_path = os.path.join(result_path, directory, 'result_total.hdf5')
            fold = 0
            for key in list(trial_mean_dict.keys()):
                general.save_hdf5(result_total_path, '{}_fold{}'.format(key, int(fold)), trial_mean_dict[key])
            
        result_total_path = os.path.join(result_path, directory, 'result_total.hdf5')
        fold_mean_dict = mean_results(result_total_path, folder_num)
        for key in list(fold_mean_dict.keys()):
            general.save_hdf5(result_total_path, key[:-5], fold_mean_dict[key])

#         print('end')
#         time.sleep(100)

            
            
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, viz, env_name='main'):
#         self.viz = Visdom(port=args.port)
        self.viz = viz
        self.env = env_name
        self.plots = {}
    def plot(self, yaxis_name, xaxis_name, log_name, directory_name, x, y):
        title_name = log_name + '_' + directory_name
        self.plots[title_name] = self.viz.line(X=x, Y=y, env=self.env, opts=dict(
            title=title_name,
            xlabel=xaxis_name,
            ylabel=yaxis_name)
        )

def visdom_plot(result_path):
    vis = Visdom(port=8081, log_to_filename='result')

    # Plot the only results that have not been plotted.
    all_directories = [directory for directory in os.listdir(os.path.join(result_path)) if directory[:2] == '20' and directory[-4:] != 'done']
    for directory in all_directories:
        if 'result_total.hdf5' in os.listdir(os.path.join(result_path, directory)):
            pass
        else:
            all_directories.pop(all_directories.index(directory))
#             print('There is no result_total.hdf5 file in {}'.format(directory))
#     print('Directories to plot :')
#     for directory in all_directories:
#         print(directory)


    for directory in all_directories:
        visplot = VisdomLinePlotter(vis, 'result_{}'.format(directory))
        with h5py.File(os.path.join(result_path, directory, 'result_total.hdf5'), 'r') as f:
            train_loss = f['train_loss'][()]
            train_class_loss = f['train_class_loss'][()]
            train_lrp_loss = f['train_lrp_loss'][()]

            val_loss = f['val_loss'][()]
            val_class_loss = f['val_class_loss'][()]
            val_lrp_loss = f['val_lrp_loss'][()]

            val_accuracy = f['val_accuracy'][()]

            x_axis = np.arange(train_loss.shape[0])
            y_train_loss = train_loss
            y_train_class_loss = train_class_loss
            y_train_lrp_loss = train_lrp_loss

            y_val_loss = val_loss
            y_val_class_loss = val_class_loss
            y_val_lrp_loss = val_lrp_loss

            y_val_accuracy = val_accuracy

            visplot.plot(yaxis_name='loss', 
                         xaxis_name='epoch', 
                         log_name='Train_loss', directory_name=directory, x=x_axis, y=y_train_loss)
            visplot.plot(yaxis_name='loss', 
                         xaxis_name='epoch', 
                         log_name='Train_class_loss', directory_name=directory, x=x_axis, y=y_train_class_loss)
            visplot.plot(yaxis_name='loss', 
                         xaxis_name='epoch', 
                         log_name='Train_lrp_loss', directory_name=directory, x=x_axis, y=y_train_lrp_loss)

            visplot.plot(yaxis_name='loss', 
                         xaxis_name='epoch', 
                         log_name='Val_loss', directory_name=directory, x=x_axis, y=y_val_loss)
            visplot.plot(yaxis_name='loss', 
                         xaxis_name='epoch', 
                         log_name='Val_class_loss', directory_name=directory, x=x_axis, y=y_val_class_loss)
            visplot.plot(yaxis_name='loss', 
                         xaxis_name='epoch', 
                         log_name='Val_lrp_loss', directory_name=directory, x=x_axis, y=y_val_lrp_loss)

            visplot.plot(yaxis_name='accuracy', 
                         xaxis_name='epoch', 
                         log_name='Val_accuracy', directory_name=directory, x=x_axis, y=y_val_accuracy)

            os.rename(os.path.join(result_path, directory), os.path.join(result_path, directory+'_done'))



def update_accuracy(result_path):
    # Plot the only results that have not been plotted.
    all_directories = [directory for directory in os.listdir(os.path.join(result_path)) if directory[:2] == '20' and directory[-4:] != 'done']
    for directory in all_directories:
        if 'result_total.hdf5' in os.listdir(os.path.join(result_path, directory)):
            os.rename(os.path.join(result_path, directory), os.path.join(result_path, directory+'_done'))
    
    all_directory = [directory for directory in os.listdir(result_path) if directory[-4:] == 'done']
    all_directory.sort()
    
    # max validation accuracy 찾아서 setting.txt에 기록
    for index, directory in enumerate(all_directory):
        with h5py.File(os.path.join(result_path, directory, 'result_total.hdf5'), 'r') as f: ####
            val_accuracy = f['val_accuracy'][()]
            val_auc = f['val_auc'][()]
            if val_accuracy.shape[0] == 0:
                max_val_accuracy = None
            else:
                max_val_accuracy = np.max(val_accuracy)
                max_val_auc = val_auc[np.argmax(val_accuracy)]
        
        setting_dict = help_execute.get_setting(os.path.join(result_path, directory))
#         if 'accuracy' not in setting_dict:
        if max_val_accuracy is None:
            pass
        else:
            setting_dict['accuracy'] = [str(np.round(max_val_accuracy, 3))]
            setting_dict['auc'] = [str(np.round(max_val_auc, 3))]
        help_execute.write_setting(setting_dict, os.path.join(result_path, directory))
    
    # max train accuracy 찾아서 setting.txt에 기록
    for index, directory in enumerate(all_directory):
        with h5py.File(os.path.join(result_path, directory, 'result_total.hdf5'), 'r') as f: ####
            train_accuracy = f['train_accuracy'][()]
            if train_accuracy.shape[0] == 0:
                max_train_accuracy = None
            else:
                max_train_accuracy = np.max(train_accuracy)
        
        setting_dict = help_execute.get_setting(os.path.join(result_path, directory))
#         if 'accuracy' not in setting_dict:
        if max_train_accuracy is None:
            pass
        else:
            setting_dict['train_accuracy'] = [str(np.round(max_train_accuracy, 3))]
        help_execute.write_setting(setting_dict, os.path.join(result_path, directory))
        
def update_accuracy_window(result_path):
    # Plot the only results that have not been plotted.
    all_directories = [directory for directory in os.listdir(os.path.join(result_path)) if directory[:2] == '20' and directory[-4:] != 'done']
    for directory in all_directories:
        if 'result_total.hdf5' in os.listdir(os.path.join(result_path, directory)):
            os.rename(os.path.join(result_path, directory), os.path.join(result_path, directory+'_done'))
    
    all_directory = [directory for directory in os.listdir(result_path) if directory[-4:] == 'done']
    all_directory.sort()
    for index, directory in enumerate(all_directory):
        max_val_accuracy = None
        with h5py.File(os.path.join(result_path, directory, 'result_total.hdf5'), 'r') as f: ####
            val_accuracy = f['val_accuracy'][()]
            if val_accuracy.shape[0] == 0:
                max_val_accuracy = None
            else:
                max_val_accuracy = np.max(val_accuracy)
        
        setting_dict = help_execute.get_setting(os.path.join(result_path, directory))
#         if 'accuracy' not in setting_dict:
        if max_val_accuracy is None:
            pass
        else:
            setting_dict['accuracy'] = [str(np.round(max_val_accuracy, 3))]
        help_execute.write_setting(setting_dict, os.path.join(result_path, directory))
        
        
def update_mIoU(result_path):
    # Plot the only results that have not been plotted.
    all_directories = [directory for directory in os.listdir(os.path.join(result_path)) if directory[:2] == '20' and directory[-4:] != 'done']

    all_directory = [directory for directory in os.listdir(result_path) if directory[-4:] == 'done']
    all_directory.sort()
    
    for index, directory in enumerate(all_directory):
        if 'mIOU_R_00.hdf5' not in os.listdir(os.path.join(result_path, directory)):
            continue
            
        mIoU = None
        with h5py.File(os.path.join(result_path, directory, 'mIOU_R_00.hdf5'), 'r') as f: ####
            IoU = []
            for key in list(f.keys()):
                if '{}/true_pos/ori/IoU'.format(key) in f:
                    IoU += list(f['{}/true_pos/ori/IoU'.format(key)][()])
                else:
                    pass
            
            if len(IoU) == 0:
                pass
            else:
                mIoU = sum(IoU) / len(IoU)
        
        setting_dict = help_execute.get_setting(os.path.join(result_path, directory))
#         if 'accuracy' not in setting_dict:
        if mIoU is None:
            pass
        else:
            setting_dict['mIoU'] = [str(np.round(mIoU, 4))]
        help_execute.write_setting(setting_dict, os.path.join(result_path, directory))
        

def update_AUC(result_path):
    # Plot the only results that have not been plotted.
    all_directories = [directory for directory in os.listdir(os.path.join(result_path)) if directory[:2] == '20' and directory[-4:] != 'done']

    all_directory = [directory for directory in os.listdir(result_path) if directory[-4:] == 'done']
    all_directory.sort()
    
    for index, directory in enumerate(all_directory):
        if 'ROC_AUC.hdf5' not in os.listdir(os.path.join(result_path, directory)):
            continue
        
        auc = None
        with h5py.File(os.path.join(result_path, directory, 'ROC_AUC.hdf5'), 'r') as f: ####
            auc = f['auc'][()]
        
        setting_dict = help_execute.get_setting(os.path.join(result_path, directory))

        if auc is None:
            pass
        else:
            setting_dict['AUC'] = [str(np.round(auc, 4))]
        help_execute.write_setting(setting_dict, os.path.join(result_path, directory))
    


    
    
def update_latest_modi_time_of_hyper_dirs():
    result_dirs = [dir for dir in os.listdir('./') if dir[:7] == 'result_']
    result_dirs.sort()
    
    modi_time_dict = collections.OrderedDict()
    modi_time_dict['latest_modification'] = []
    for result_dir in result_dirs:
        hyper_dirs = [dir for dir in os.listdir(result_dir) if dir[:2] == '20']
        hyper_dirs.sort()

        for hyper_dir in hyper_dirs:
            hyper_path = os.path.join(result_dir, hyper_dir)
            result_files = [file for file in os.listdir(os.path.join(hyper_path)) if 'result' in file]
            
            setting_dict = help_execute.get_setting(hyper_path)
            if len(result_files) == 0:
                setting_dict['latest_modification'] = ['0000-00-00 00:00:00']
            else:
                mtimes = [os.path.getmtime(os.path.join(hyper_path, file)) for file in result_files]
                latest_modi_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max(mtimes)))
                setting_dict['latest_modification'] = [latest_modi_time]
            help_execute.write_setting(setting_dict, os.path.join(hyper_path))

            
def set_verified(hyper_dict):
    result_dirs = [dir for dir in os.listdir('./') if dir[:7] == 'result_']
    result_dirs.sort()
    
    verified_dict = collections.OrderedDict()
    verified_dict['verified'] = []
    for result_dir in result_dirs:
        hyper_dirs = [dir for dir in os.listdir(result_dir) if dir[:2] == '20']
        hyper_dirs.sort()

        for hyper_dir in hyper_dirs:
            hyper_path = os.path.join(result_dir, hyper_dir)
            result_files = [file for file in os.listdir(os.path.join(hyper_path)) if 'result' in file]
            
            setting_dict = help_execute.get_setting(hyper_path)
            
            hyper_dict_done_version = [hyper+'_done' for hyper in hyper_dict.keys()]
            if hyper_dir in hyper_dict:
                print('{} modified'.format(hyper_dir))
                setting_dict['verified'] = [hyper_dict[hyper_dir]]
                help_execute.write_setting(setting_dict, os.path.join(hyper_path))
            elif hyper_dir in hyper_dict_done_version:
                print('{} modified'.format(hyper_dir))
                setting_dict['verified'] = [hyper_dict[hyper_dir[:-5]]]
                help_execute.write_setting(setting_dict, os.path.join(hyper_path))
            

            
def check_current_fold_n_trial():
    result_dirs = [dir for dir in os.listdir('./') if dir[:7] == 'result_']
    result_dirs.sort()
    
    process = [(int(p), c) for p, c in [x.rstrip('\n').split(' ', 1) for x in os.popen('ps h -eo pid:1,command')]]
    
    verified_dict = collections.OrderedDict()
    verified_dict['verified'] = []
    for result_dir in result_dirs:
        hyper_dirs = [dir for dir in os.listdir(result_dir) if dir[:2] == '20']
        hyper_dirs.sort()
        
        for hyper_dir in hyper_dirs:
            hyper_path = os.path.join(result_dir, hyper_dir)
            result_files = [file for file in os.listdir(os.path.join(hyper_path)) if 'result' in file and len(list(filter(str.isdigit, file[:-5]))) > 0]
            result_files.sort()
            
            cumul_fold = len(result_files) - 1
            cumul_trial = 0
            current_trial = 0
            
            for num, file in enumerate(result_files):
                if len(result_files) - 1 == num:
                    with h5py.File(os.path.join(hyper_path, file), 'r') as f:
                        val_accuracy_keys = [key for key in list(f.keys()) if 'val_accuracy' in key]
                        val_accuracy_keys.sort()
                        current_trial = len(val_accuracy_keys)
                else:
                    with h5py.File(os.path.join(hyper_path, file), 'r') as f:
                        val_accuracy_keys = [key for key in list(f.keys()) if 'val_accuracy' in key]
                        val_accuracy_keys.sort()
                        cumul_trial += len(val_accuracy_keys)
                        
            setting_dict = help_execute.get_setting(hyper_path)
            setting_dict['fold_n_trial'] = ['{}+1 / {}+{}'.format(cumul_fold, cumul_trial, current_trial)]
            help_execute.write_setting(setting_dict, os.path.join(hyper_path))

            
def get_parent_pid_by_name(pid, parent_name):
    p = psutil.Process(pid)
    parent_pid = '-'
    for i in range(10):
        p = psutil.Process(p.ppid())
        if p.name() == parent_name:
            parent_pid = p.pid
        elif p.name() == 'systemd':
            break
    return parent_pid
            

def check_process_running():
    result_dirs = [dir for dir in os.listdir('./') if dir[:7] == 'result_']
    result_dirs.sort()
    
    process = [(int(p), c) for p, c in [x.rstrip('\n').split(' ', 1) for x in os.popen('ps h -eo pid:1,command')]]
    
    verified_dict = collections.OrderedDict()
    verified_dict['verified'] = []
    for result_dir in result_dirs:
        hyper_dirs = [dir for dir in os.listdir(result_dir) if dir[:2] == '20']
        hyper_dirs.sort()

        for hyper_dir in hyper_dirs:
            exist = False
            hyper_path = os.path.join(result_dir, hyper_dir)
            for pid, name in process:
                if 'sudo -S python3 main' in name and hyper_dir in name:
                    exist = True
                    
                    setting_dict = help_execute.get_setting(hyper_path)
                    setting_dict['pid'] = [str(pid)]
                    
                    screen_pid = get_parent_pid_by_name(pid, 'screen')
                    
                    setting_dict['screen_pid'] = [str(screen_pid)]
                    help_execute.write_setting(setting_dict, os.path.join(hyper_path))
                    
            if exist == False:
                setting_dict = help_execute.get_setting(hyper_path)
                setting_dict['pid'] = ['-']
                help_execute.write_setting(setting_dict, os.path.join(hyper_path))
                    

def linux_command(command):
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    while True:
        output = p.stdout.readline()
        print(output.strip().decode("utf-8"))
        # Do something else
        return_code = p.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output 
            for output in p.stdout.readlines():
                print(output.strip())
                
            output, err = p.communicate(b"input data that is passed to subprocess' stdin")
            print(err.decode("utf-8"))
            break

            
def rsync2remote(remote_ip, key_path, copying_object, target_path, exclude_list=None):
    command = [
        "rsync", 
        "-avhe", 
        'ssh -i {} -p 2048'.format(key_path),
        "{}".format(copying_object),
        "juna@{}:{}".format(remote_ip, target_path)
    ]
    
    if exclude_list != None:
        new_exclude_list = []
        for element in exclude_list:
            new_exclude_list += ['--exclude', element]
        command = command + new_exclude_list
    
    linux_command(command)


def execute_remote_sh(remote_ip, target_file):
    command = ['sshpass', '-p', 'asdf', 
               'ssh', 'juna@{}'.format(remote_ip), '-p2048', 
               target_file]
    linux_command(command)


def rsync2local(remote_ip, key_path, copying_object, target_path, exclude_list=None):
    command = [
        "rsync",
        "-avzhe",
        'ssh -i {} -p 2048'.format(key_path),
        "juna@{}:{}".format(remote_ip, copying_object), 
        "{}".format(target_path)
    ]
    
    if exclude_list != None:
        new_exclude_list = []
        for element in exclude_list:
            new_exclude_list += ['--exclude', element]
        command = command + new_exclude_list
    
    linux_command(command)
            
            
def update_remote_server_result(remote_ip, key_path='/media/volume1/juna/.ssh/id_rsa'):
    
    copying_object = '/media/volume1/juna/LRP_project'
    target_path = '/home/juna/'
    exclude_list = ['data*', 'result_*-*_*', 'result_relevance_*', '.git', 'data', 'testing_place', 'activmask', 'project_tmp', 'git_repos']
    rsync2remote(remote_ip, key_path, copying_object, target_path, exclude_list)
    
    target_file = '/home/juna/LRP_project/project_brain_lrp_train/update_remote_server_result.sh'
    execute_remote_sh(remote_ip, target_file)
    
    exclude_list = ['mIOU_*']
    copying_object = '/home/juna/LRP_project/project_brain_lrp_train/result_*-*_*'
    target_path = '/media/volume1/juna/LRP_project/project_brain_lrp_train'
    rsync2local(remote_ip, key_path, copying_object, target_path, exclude_list)
    
            
            
            

def result2dataframe(result_path, result_path_old=None):
    frame_dict = collections.OrderedDict()
    
    result_dirs = [os.path.join(result_path, dir) for dir in os.listdir(result_path) if dir[:7] == 'result_']
    if result_path_old != None:
        result_dirs_old = [os.path.join(result_path_old, dir) for dir in os.listdir(os.path.join(result_path, result_path_old)) if dir[:7] == 'result_']
        result_dirs += result_dirs_old
    result_dirs.sort()

    count = 0
    for result_dir in result_dirs:
        hyper_dirs = [dir for dir in os.listdir(result_dir) if dir[:2] == '20']
        hyper_dirs.sort()

        for hyper_dir in hyper_dirs:
            count += 1
            hyper_path = os.path.join(result_dir, hyper_dir)

            setting_dict = help_execute.get_setting(hyper_path)

            value = []
            num_of_previous_values = 0

            if len(frame_dict) == 0:
                frame_dict = setting_dict
            else:
                num_of_previous_values = len(frame_dict[list(frame_dict.keys())[0]])
                for key in setting_dict:
                    if key not in frame_dict:
                        frame_dict[key] = ['-' for i in range(num_of_previous_values)]
                        frame_dict[key] += setting_dict[key]
                    else:
                        frame_dict[key] += setting_dict[key]
                for key in frame_dict.keys():
                    if key not in setting_dict.keys():
                        frame_dict[key] += ['-']

        for key in frame_dict.keys():
            frame_dict[key] += ['-']


    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', -1)
    
    return pd.DataFrame(frame_dict)
            
            
def dataframe_edit(df, priorities):
    columns = list(df.columns)
    others = []
    for hyper in columns:
        if hyper not in priorities:
            others.append(hyper)
    columns = priorities + others

    df = df.reindex(columns=columns)
    df.insert(len(priorities), "-", ['||||||' for i in range(df.shape[0])], True)
    return df
          
    
def get_hyper_path_list(project_path='/media/volume1/juna/LRP_project/project_brain_lrp_train', old_result_path='before_20191001'):       
    hyper_path_list = []
    resultdir_path_list = [os.path.join(project_path, result_dir) for result_dir in fnmatch.filter(os.listdir(project_path), 'result_*')]
    if old_result_path != None:
        project_path = os.path.join(project_path, old_result_path)
        resultdir_path_list += [os.path.join(project_path, result_dir) for result_dir in fnmatch.filter(os.listdir(project_path), 'result_*')]

    for result_path in resultdir_path_list:
        hyper_path_list += [os.path.join(result_path, hyper_dir) for hyper_dir in fnmatch.filter(os.listdir(result_path), '20*')]
    hyper_path_list.sort()
    return hyper_path_list


def multiplot(hyper_path_list, plot_list=['train_loss','train_class_loss','train_lrp_loss','val_loss','val_class_loss','val_lrp_loss','val_accuracy'], figsize=(50, 5)):
    for hyper_path in hyper_path_list:
        if 'result_total.hdf5' in os.listdir(hyper_path):
            with h5py.File(os.path.join(hyper_path, 'result_total.hdf5'), 'r') as f:
    #             print(list(f.keys()))

                w, h = figaspect(1/len(plot_list)) * 0.36*len(plot_list)
                fig, ax = plt.subplots(nrows=1, ncols=len(plot_list), figsize=(w, h)) # (40.32, 5.759)

                title = os.path.basename(hyper_path)
                fig.suptitle('{}      {}      {}'.format(title, title, title), fontsize=20)

                for num, plot_name in enumerate(plot_list):
                    y = f[plot_name][()]
                    x = np.arange(y.shape[0])
                    ax[num].plot(x, y)
                    ax[num].set_title(plot_name)
                    ax[num].set_xlabel('epoch')
                    ax[num].set_aspect(1/ax[num].get_data_ratio())
                plt.show()
        else:
            print('no result_total.hdf5 in {}'.format(hyper_path))



            

# def merge_plot(hyper_path_list, plot_list=['train_loss','train_class_loss','train_lrp_loss','val_loss','val_class_loss','val_lrp_loss','val_accuracy'], figsize=(50, 5)):
    
#     w, h = figaspect(1/len(plot_list)) * 0.36*len(plot_list)
#     fig, ax = plt.subplots(nrows=1, ncols=len(plot_list), figsize=(w, h)) # (40.32, 5.759)
    
#     for hyper_path in hyper_path_list:
#         if 'result_total.hdf5' in os.listdir(hyper_path):
#             with h5py.File(os.path.join(hyper_path, 'result_total.hdf5'), 'r') as f:
#     #             print(list(f.keys()))
                
#                 for num, plot_name in enumerate(plot_list):
#                     y = f[plot_name][()]
#                     x = np.arange(y.shape[0])
#                     ax[num].plot(x, y, label=os.path.basename(hyper_path).replace('_done','')[-4:])
#                     ax[num].set_title(plot_name)
#                     ax[num].set_xlabel('epoch')
#                     ax[num].set_aspect(1/ax[num].get_data_ratio())
#                     ax[num].legend(loc="upper left")
#         else:
#             print('no result_total.hdf5 in {}'.format(hyper_path))
#     plt.show()


def merge_plot(hyper_path_list, plot_list=['train_loss','train_class_loss','train_lrp_loss','train_accuracy','val_loss','val_class_loss','val_lrp_loss','val_accuracy'], figsize=(50, 5), label_list=None, moving_ave=None):
    if label_list != None and label_list != 'No label' and len(hyper_path_list) != len(label_list):
        print('len(hyper_path_list) : {} | len(label_list) : {}'.format(len(hyper_path_list), len(label_list)))
        raise Exception('len(hyper_path_list) is different from len(label_list)')
    
    w, h = figaspect(1/len(plot_list)) * 0.36*len(plot_list)
    fig, ax = plt.subplots(nrows=1, ncols=len(plot_list), figsize=(w, h)) # (40.32, 5.759)
    
    for hyper_num, hyper_path in enumerate(hyper_path_list):
        if 'result_total.hdf5' in os.listdir(hyper_path):
            with h5py.File(os.path.join(hyper_path, 'result_total.hdf5'), 'r') as f:
                print(list(f.keys()))
                
                train_val_accuracy = None
                for num, plot_name in enumerate(plot_list):
                    if plot_name != 'train_val_accuracy':
                        y = f[plot_name][()][:30]
                    else:
                        y = train_val_accuracy
                    x = np.arange(y.shape[0])
                    
                    if moving_ave != None:
                        if (moving_ave % 2) != 1:
                            raise Exception('moving_ave is not an odd number')
                        padding_num = int((moving_ave-1) / 2)
                        y = np.array([np.nan for i in range(padding_num)] + list(y) + [np.nan for i in range(padding_num)])

                        new_y = []
                        for i in range(y.shape[0]-(moving_ave-1)):
                            window = y[i:i+moving_ave]
#                             print(window)
#                             print(np.isnan(window))
#                             time.sleep(100)
                            new_y.append(window[np.logical_not(np.isnan(window))].mean())
                        y = np.array(new_y)
                    
                    if 'train_val_accuracy' in plot_list and (plot_name == 'train_accuracy' or plot_name == 'val_accuracy'):
                        if train_val_accuracy is None:
                            train_val_accuracy = np.zeros_like(y)
                        train_val_accuracy += y
            
                    if label_list == None:
                        ax[num].plot(x, y, label=os.path.basename(hyper_path).replace('_done','')[-4:])
                    elif label_list == 'No label':
                        ax[num].plot(x, y)
                    else:
                        if plot_name == 'val_accuracy':
                            print('{} : {}'.format(label_list[hyper_num], y.max()))
                        ax[num].plot(x, y, label=label_list[hyper_num])
                    ax[num].set_title(plot_name, fontsize=20)
                    ax[num].set_xlabel('epoch')
                    ax[num].set_aspect(1/ax[num].get_data_ratio())
                    if label_list == 'No label':
                        pass
                    else:
                        ax[num].legend(fontsize=10) # loc="upper right"
        else:
            print('no result_total.hdf5 in {}'.format(hyper_path))
    plt.show()
    
    

def show_merge_plot(hyper_path_list, plot_hyper, df, plot_list, label_list=None, moving_ave=None):
    merging_hypers = []
    merging_hypers_full_path = []
    
    for hyper in plot_hyper:
        if len(fnmatch.filter(hyper_path_list, hyper)) > 1:
            raise Exception('selected hyper [{}] path is more than 1'.format(fnmatch.filter(hyper_path_list, hyper)))
        merging_hypers_full_path += fnmatch.filter(hyper_path_list, hyper)
    
    merging_hypers_string = [os.path.basename(hyper).replace('_done','') for hyper in merging_hypers_full_path]
    print(merging_hypers_string)
    display(df.loc[df['hyper_dir'].isin(merging_hypers_string)])

    merge_plot(merging_hypers_full_path, plot_list, label_list=label_list, moving_ave=moving_ave)
    
    
def merge_one_plot(hyper_path_list, plot_list=['train_loss','train_class_loss','train_lrp_loss','val_loss','val_class_loss','val_lrp_loss','val_accuracy'], figsize=(50, 5)):
    
    w, h = figaspect(1/len(plot_list)) * 0.36*len(plot_list)
    
    for hyper_path in hyper_path_list:
        if 'result_total.hdf5' in os.listdir(hyper_path):
            with h5py.File(os.path.join(hyper_path, 'result_total.hdf5'), 'r') as f:
    #             print(list(f.keys()))
                
                for num, plot_name in enumerate(plot_list):
                    y = f[plot_name][()]
                    x = np.arange(y.shape[0])
                    plt.plot(x, y)
                    plt.set_title(plot_name)
                    plt.set_xlabel('epoch')
                    plt.set_aspect(1/ax[num].get_data_ratio())
        else:
            print('no result_total.hdf5 in {}'.format(hyper_path))
    plt.show()
            
            
            
            
def merge_results(project_path, hyper1, hyper2):
    hyper_path_list = get_hyper_path_list(project_path)

    # for i in hyper_path_list:
    #     print(i)

    hyper1 = fnmatch.filter(hyper_path_list, hyper1)
    hyper2 = fnmatch.filter(hyper_path_list, hyper2)

    if len(hyper1) != 1 or len(hyper2) != 1:
        raise Exception('length is not 1')
    else:
        hyper1 = hyper1[0]
        hyper2 = hyper2[0]
#         print(hyper1)
#         print(hyper2)

    print(os.listdir(hyper1))
    print(os.listdir(hyper2))

    result_path1 = os.path.split(hyper1)[0]
    result_path2 = os.path.split(hyper2)[0]

    latter_result_path = max(result_path1, result_path2)
#     print(latter_result_path, '@@@@@')
    result_folder_front = '_'.join(os.path.basename(latter_result_path).split('_')[:2])


    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    new_result_path = os.path.join(os.path.split(latter_result_path)[0], result_folder_front+'_'+time)


    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    new_hyper_path = os.path.join(new_result_path, time)
    print('new_hyper_path :', new_hyper_path)
    os.makedirs(new_hyper_path) # trial : 5, screen_pid : '-', result_dir : ~~, hyper_dir : '~~'

    result1_dict = collections.OrderedDict()
    result2_dict = collections.OrderedDict()
    new_result_dict = collections.OrderedDict()

    with h5py.File(os.path.join(hyper1, 'result00.hdf5'), 'r') as f:
#         print(list(f.keys()))
        for key in list(f.keys()):
            result1_dict[key] = f[key][()]

    with h5py.File(os.path.join(hyper2, 'result00.hdf5'), 'r') as f:
#         print(list(f.keys()))
        for key in list(f.keys()):
            result2_dict[key] = f[key][()]

#     print()

    list_of_trial1 = []
    list_of_key = []

    length = -1
    for key in result1_dict.keys():
        list_of_trial1.append(int(''.join(filter(str.isdigit, key))))
        list_of_key.append(''.join(i for i in key if not i.isdigit()))
        print(key, result1_dict[key].shape, result1_dict[key][()].mean())

        list_of_trial1
        if length == -1:
            length = len(result1_dict[key])
        else:
            if length != len(result1_dict[key]):
                raise Exception('length is different')

    list_of_trial1 = list(set(list_of_trial1))

    if length == 0:
        raise Exception('length is 0!!')

    print()
    list_of_trial2 = []
    for key in result2_dict.keys():
        list_of_trial2.append(int(''.join(filter(str.isdigit, key))))
        print(key, result2_dict[key].shape, result2_dict[key][()].mean())

        if length != len(result2_dict[key]):
            raise Exception('length is different')

    list_of_trial2 = list(set(list_of_trial2))

    list_of_key.sort()
#     print(list_of_key)
#     print(list_of_trial1)
#     print(list_of_trial2)

    total_trial = len(list_of_trial1) + len(list_of_trial2)


    for new_key in list_of_key:
        count = 0
        for key in result1_dict.keys():
            if new_key in key:
                new_result_dict[new_key+str(count)] = result1_dict[key]
                count += 1

        for key in result2_dict.keys():
            if new_key in key:
                new_result_dict[new_key+str(count)] = result2_dict[key]
                count += 1
    print()

    with h5py.File(os.path.join(new_hyper_path, 'result00.hdf5'), 'w') as f:
        for key in new_result_dict.keys():
#             print(key, new_result_dict[key].shape, new_result_dict[key].mean())
            f.create_dataset(key, data=new_result_dict[key])

    print()

    with h5py.File(os.path.join(new_hyper_path, 'result00.hdf5'), 'r') as f:
        for key in f.keys():
            print(key, f[key].shape, new_result_dict[key][()].mean())

            

            
def show_one_img(data_path, which_fold_to_show, img_key, num, img_color):
    depth = None
    with h5py.File(os.path.join(data_path, 'so_ph_{:02d}.hdf5'.format(which_fold_to_show)), 'r') as g:
        shape = g[img_key][num].shape
        if len(shape) == 3:
            depth = shape[0]
            if depth == 3:
                if img_color:
                    img = cv2.cvtColor(np.transpose(g[img_key][num], (1, 2, 0)), cv2.COLOR_BGR2RGB)
                else:
                    img = np.transpose(g[img_key][num], (1, 2, 0))
            elif depth == 1:
                img = g[img_key][num][0]
            else:
                raise Exception('data depth is not 3 or 1')
        else:
            img = g[img_key][num]
    
    if img_color:
        plt.imshow(img)
        plt.show()
    else:
        general.imshow_depth(img)
        
def show_img_mask_relevance_for_relevance_map_folder(relevance_path, hyper_list, which_fold_to_show, ori_df, img_start_num, num_of_showing_img):
    for hyper in hyper_list:
        selected_hyper = help_execute.get_hyperdir_by_setting(relevance_path, 'test_model_path', hyper)
        if selected_hyper == None:
            raise Exception('There is no such hyper_dir setting file that has {} for test_model_path'.format(hyper))
            
        setting_dict = help_execute.get_setting(selected_hyper)
        ori_data_path = setting_dict['data_path'][0]

        display(ori_df.loc[ori_df['hyper_dir'].isin([hyper])])

        with h5py.File(os.path.join(selected_hyper, 'relevance_map_{:02d}.hdf5'.format(which_fold_to_show)), 'r') as f:

            relevance_img_total_num = f['R'].shape[0]
            if relevance_img_total_num <= img_start_num or relevance_img_total_num <= img_start_num+num_of_showing_img:
                raise Exception('out of range | relevance_img_total_num is {}'.format(relevance_img_total_num))
            
            for num in range(img_start_num, img_start_num+num_of_showing_img):
                print('output : {} | label : {}'.format(np.argmax(f['output'][num]), f['label'][num]))

                show_one_img(ori_data_path, which_fold_to_show, 'test_data', num, img_color=False)
                show_one_img(ori_data_path, which_fold_to_show, 'test_mask', num, img_color=False)
                general.imshow_depth_seaborn(np.transpose(f['R'][num], (1,2,0)))
                print()
                print()
            

def show_img_mask_relevance(hyper_path_list, hyper_list, which_fold_to_show, ori_df, img_start_num, num_of_showing_img):
    for hyper in hyper_list:
        selected_hyper = fnmatch.filter(hyper_path_list, hyper)
        if len(selected_hyper) == 0:
            raise Exception('There is no such hyper_dir named {}'.format(hyper))
        elif len(selected_hyper) > 1:
            raise Exception('There are more than one {} path'.format(hyper))
        selected_hyper = selected_hyper[0]

        setting_dict = help_execute.get_setting(selected_hyper)
        ori_data_path = setting_dict['data_path'][0]

        display(ori_df.loc[ori_df['hyper_dir'].isin([os.path.basename(selected_hyper).replace('_done', '')])])

        with h5py.File(os.path.join(selected_hyper, 'relevance_map_{:02d}.hdf5'.format(which_fold_to_show)), 'r') as f:

            relevance_img_total_num = f['R'].shape[0]
            if relevance_img_total_num <= img_start_num or relevance_img_total_num <= img_start_num+num_of_showing_img:
                raise Exception('out of range | relevance_img_total_num is {}'.format(relevance_img_total_num))

            for num in range(img_start_num, img_start_num+num_of_showing_img):
                print('output : {} | label : {}'.format(np.argmax(f['output'][num]), f['label'][num]))

    #             result_process.show_one_img(ori_data_path, which_fold_to_show, 'test_data', num, img_color=False)
    #             reshaped_R = f['R'][num]
    #             print(reshaped_R.shape)
    #             seaborn.heatmap(reshaped_R, cmap='RdBu_r', center=0, square=True, vmin=-2.5, vmax=2.5)
    #             plt.show()
    #             print()
    #             print()
                show_one_img(ori_data_path, which_fold_to_show, 'test_data', num, img_color=False)
                show_one_img(ori_data_path, which_fold_to_show, 'test_mask', num, img_color=False)
                general.imshow_depth_seaborn(np.transpose(f['R'][num], (1,2,0)))
                print()
                print()
            
            
def show_roc(hyper_path_list, plot_hyper, df, label_list):
    plt.figure(figsize=(5,5))
    plt.plot([0, 1], [0, 1], c='r', ls='--')
    for num, hyper in enumerate(plot_hyper):
        selected_hyper = fnmatch.filter(hyper_path_list, hyper)
        
        if len(selected_hyper) == 0:
            raise Exception('There is no such hyper_dir named {}'.format(hyper))
        elif len(selected_hyper) > 1:
            raise Exception('There are more than one {} path'.format(hyper))
        selected_hyper = selected_hyper[0]
        
        display(df.loc[df['hyper_dir'].isin([os.path.basename(selected_hyper).replace('_done', '')])])

        with h5py.File(os.path.join(selected_hyper, 'ROC_AUC.hdf5'), 'r') as f:
            roc_points = f['roc'][()]
            x, y = roc_points[:,0], roc_points[:,1]
            print('share :', x.shape, y.shape)
            plt.plot(x, y, label='{}'.format(label_list[num]))
            plt.xlim(left=0, right=1)
            plt.ylim(bottom=0, top=1)
            plt.xlabel('False Positive Rate', fontsize=15)
            plt.ylabel('True Positive Rate', fontsize=15)
            plt.legend(fontsize=12)
    plt.show()
    
    
def get_selected_hyper(hyper_path_list, hyper):
    selected_hyper = fnmatch.filter(hyper_path_list, hyper)
    if len(selected_hyper) == 0:
        raise Exception('There is no such hyper_dir named {}'.format(hyper))
    elif len(selected_hyper) > 1:
        raise Exception('There are more than one {} path'.format(hyper))
    selected_hyper = selected_hyper[0]
    
    return selected_hyper
            
def select_hyper_from_the_list(hyper_path_list, hyper):
    selected_hyper = fnmatch.filter(hyper_path_list, hyper)
    if len(selected_hyper) == 0:
        raise Exception('There is no such hyper_dir named {}'.format(hyper))
    elif len(selected_hyper) > 1:
        raise Exception('There are more than one {} path'.format(hyper))
    selected_hyper = selected_hyper[0]
    return selected_hyper
    
def get_output_matrix(filename, total_perturb_num, do_softmax=False):
    output_m = []
    output_mask = None
    with h5py.File(filename, 'r') as f:
        img_num = 0
        label = f['label'][()]
        output_mask = f['output_mask'][()]
        for perturb_num in range(total_perturb_num):
            output = f['output_{:02d}'.format(perturb_num)][()]
            if do_softmax:
                output = softmax(output, axis=1)
                
#             output_m.append(output[np.arange(*label.shape), [label]].T)
            output_m.append((output[:,1])[:,np.newaxis])

    output_m = np.concatenate(output_m, axis=1)

    return output_m, output_mask

def get_images(filename, total_perturb_num, data_num):
    total_images = []
    with h5py.File(filename, 'r') as f:
        for perturb_num in range(total_perturb_num):
            images = f['image_{:02d}'.format(perturb_num)][()]
            total_images.append(np.expand_dims(images[data_num], axis=0))
    
    total_images = np.concatenate(total_images, axis=0)
    return total_images
    
def aopc_debug(filename, total_perturb_num):
    with h5py.File(filename, 'r') as f:
        print(list(f.keys()))
        print(f['R'].shape)
        
        for img_num in range(f['R'].shape[0]):
            print('################################################### image {:02d} ######################################################'.format(img_num))
            label = f['label'][img_num] # label에는 scalar가 들어감
            output = f['output'][img_num]
            print('label :', label)
            print('output :', output)
            if np.argmax(output) == label:
                print('True Positive')
            else:
                print('Not True Positive!!!!')
            R = f['R'][img_num]
            general.imshow_depth(np.transpose(R, (1, 2, 0)))
            img = f['images'][img_num]
            general.imshow_depth(np.transpose(img, (1, 2, 0)))
            print('============== perturb images ==============')
                
            for perturb_num in range(total_perturb_num-2, total_perturb_num):
                img = f['image_{:02d}'.format(perturb_num)][img_num]
                output = f['output_{:02d}'.format(perturb_num)][img_num]

                print('perturb_num :', perturb_num)
                print('label :', label)
                print('output :', output)
                if np.argmax(output) == label:
                    print('True Positive')
                else:
                    print('Not True Positive!!!!')
                
                general.imshow_depth(np.transpose(img, (1, 2, 0)))
                
        
def show_aopc_each_img(hyper_path_list, plot_hyper, df, label_list, showing_img_list, do_softmax=False):
    for data_num in showing_img_list:
        print('########################################### image {:02d} ###########################################'.format(data_num))
        for hyper_order, hyper in enumerate(plot_hyper):
            selected_hyper = select_hyper_from_the_list(hyper_path_list, hyper)

            setting_dict = help_execute.get_setting(selected_hyper)
            ori_data_path = setting_dict['data_path'][0]
            total_perturb_num = int(setting_dict['total_perturb_num'][0])

            display(df.loc[df['hyper_dir'].isin([os.path.basename(selected_hyper).replace('_done', '')])])

            images = get_images(os.path.join(selected_hyper, 'perturbed_result.hdf5'), total_perturb_num, data_num)
            images = images[:,0,:,:] # depth 3개중에 첫번째것만 사용 # shape (50,3,224,224) -> (50,224,224)

            num_of_showing_perturb = 10
            # total_perturb_num개의 perturb image들중에서 일정구간으로 뽑아서 그것들만 show
            perturb_list = list(np.round(np.linspace(0, images.shape[0]-1, num=num_of_showing_perturb)).astype(np.int)) 
            
            fig, ax = plt.subplots(nrows=1, ncols=num_of_showing_perturb, figsize=(40, 20))
            for num, perturb_num in enumerate(perturb_list):
                ax[num].imshow(images[perturb_num,:,:], cmap='gray')
            plt.show()


        for hyper_order, hyper in enumerate(plot_hyper):
            selected_hyper = select_hyper_from_the_list(hyper_path_list, hyper)

            setting_dict = help_execute.get_setting(selected_hyper)
            ori_data_path = setting_dict['data_path'][0]
            total_perturb_num = int(setting_dict['total_perturb_num'][0])


            output_m, output_mask = get_output_matrix(os.path.join(selected_hyper, 'perturbed_result.hdf5'), total_perturb_num, do_softmax)
            # 모든 output중에 image가 저장된 output만 발라냄
            sampled_output_m = output_m[output_mask]

            y = sampled_output_m[data_num]
            x = np.arange(y.shape[0])
            plt.plot(x, y, label='{}'.format(label_list[hyper_order]))
            plt.xlabel('num of perturbation', fontsize=15)
            plt.ylabel('score', fontsize=15)
            plt.legend(fontsize=15)
        plt.show()
        
        
        
    
        
        
def show_aopc(hyper_path_list, plot_hyper, df, label_list, do_softmax=False, mean_score=False, debug=False):
    fig = plt.figure(figsize=(6, 6))
    
    if mean_score:
        for hyper_order, hyper in enumerate(plot_hyper):
            selected_hyper = select_hyper_from_the_list(hyper_path_list, hyper)

            setting_dict = help_execute.get_setting(selected_hyper)
            ori_data_path = setting_dict['data_path'][0]
            total_perturb_num = int(setting_dict['total_perturb_num'][0])

            display(df.loc[df['hyper_dir'].isin([os.path.basename(selected_hyper).replace('_done', '')])])

            if debug:
                aopc_debug(os.path.join(selected_hyper, 'perturbed_result.hdf5'), total_perturb_num)
            output_m, _ = get_output_matrix(os.path.join(selected_hyper, 'perturbed_result.hdf5'), total_perturb_num, do_softmax)
            output_mean = np.mean(output_m, axis=0)

            x = np.arange(total_perturb_num)
            y = output_mean
            plt.plot(x, y, label='{}'.format(label_list[hyper_order]))
            plt.xlabel('num of perturbation', fontsize=15)
            plt.ylabel('score', fontsize=15)
            plt.legend(fontsize=12)
        plt.show()

    
    for hyper_order, hyper in enumerate(plot_hyper):
        selected_hyper = select_hyper_from_the_list(hyper_path_list, hyper)

        setting_dict = help_execute.get_setting(selected_hyper)
        ori_data_path = setting_dict['data_path'][0]
        total_perturb_num = int(setting_dict['total_perturb_num'][0])
        
        if not mean_score:
            display(df.loc[df['hyper_dir'].isin([os.path.basename(selected_hyper).replace('_done', '')])])
        
        if debug:
            aopc_debug(os.path.join(selected_hyper, 'perturbed_result.hdf5'), total_perturb_num)
        output_m, _ = get_output_matrix(os.path.join(selected_hyper, 'perturbed_result.hdf5'), total_perturb_num, do_softmax)
        output_mean = np.mean(output_m, axis=0)
        aopc = np.cumsum(output_mean[0] - output_mean)
            
        x = np.arange(total_perturb_num)
        y = aopc
        plt.plot(x, y, label='{}'.format(label_list[hyper_order]))
#         plt.xlim(right=10)
#         plt.ylim(top=25)
        plt.xlabel('num of perturbation', fontsize=15)
        plt.ylabel('AOPC', fontsize=15)
        plt.legend(fontsize=12)
    plt.show()
    
    

def show_IoU(hyper_list, hyper_path_list, confusion):
    assert confusion in ['true_pos', 'true_neg', 'false_pos', 'false_neg']
    for hyper in hyper_list:
        selected_hyper = fnmatch.filter(hyper_path_list, hyper)
        if len(selected_hyper) == 0:
            raise Exception('There is no such hyper_dir named {}'.format(hyper))
        elif len(selected_hyper) > 1:
            raise Exception('There are more than one {} path'.format(hyper))
        selected_hyper = selected_hyper[0]

        print(selected_hyper)

        with h5py.File(os.path.join(selected_hyper, 'mIOU_R_00.hdf5'), 'r') as f:
            print(list(f['0'].keys()))
            print('img_{}.shape :'.format(confusion), f['0/{}/ori/img'.format(confusion)].shape)
            print('R_{}.shape :'.format(confusion), f['0/{}/ori/R'.format(confusion)].shape)
            print('mask_{}.shape :'.format(confusion), f['0/{}/ori/mask'.format(confusion)].shape)
            
#             for img_num in range(f['R_true_pos'].shape[0]):
#                 img = f['img_true_pos'][img_num][[0]] # 첫번째 슬라이스만 get함
#                 R = f['R_true_pos'][img_num]
                
            for img_num in range(f['0/{}/ori/R'.format(confusion)].shape[0]):
                img = f['0/{}/ori/img'.format(confusion)][img_num][[0]] # 첫번째 슬라이스만 get함
                R = f['0/{}/ori/R'.format(confusion)][img_num]
                
                print('R.shape :', R.shape)
                R = skimage.transform.resize(R, (R.shape[0], 224, 224))
                mask = f['0/{}/ori/mask'.format(confusion)][img_num]
                img_idx = f['0/{}/ori/img_idx'.format(confusion)][img_num]
                if confusion == 'true_pos':
                    union = f['0/{}/ori/union'.format(confusion)][img_num]
                    IS = f['0/{}/ori/IS'.format(confusion)][img_num]
                    IoU = f['0/{}/ori/IoU'.format(confusion)][img_num]

                print('============================ {} ============================'.format(img_idx))
                if confusion == 'true_pos':
                    print('mIoU : ', f['0/{}/ori/IoU'.format(confusion)][()].mean())
    #             print(f['IoU'][()])

                print('img_{}'.format(confusion))
                if img.shape[0] == 1:
                    plt.imshow(img[0], cmap='gray')
                    plt.axis('off')
                    plt.show()
                else:
                    general.imshow_depth(np.transpose(img, (1,2,0)))

                print('mask_modi_{}'.format(confusion))
                if mask.shape[0] == 1:
                    plt.imshow(mask[0], cmap='gray')
                    plt.axis('off')
                    plt.show()
                else:
                    general.imshow_depth(np.transpose(mask, (1,2,0)))

                print('R_{}'.format(confusion))
                if R.shape[0] == 1:
                    seaborn.heatmap(R[0], cmap='RdBu_r', center=0, cbar=False, square=True) # , ax=ax[depth_num], cbar_kws={'shrink': .2}
                    plt.axis('off')    
                    plt.show()
                else:
                    general.imshow_depth_seaborn(np.transpose(R, (1,2,0)), cbar=False)
                
                if confusion == 'true_pos':
                    print('union')
                    if union.shape[0] == 1:
                        plt.imshow(union[0], cmap='gray')
                        plt.axis('off')
                        plt.show()
                    else:
                        general.imshow_depth(np.transpose(union, (1,2,0)))

                    print('IS')
                    if IS.shape[0] == 1:
                        plt.imshow(IS[0], cmap='gray')
                        plt.axis('off')
                        plt.show()
                    else:
                        general.imshow_depth(np.transpose(IS, (1,2,0)))


                    print('IoU :', IoU)

    #             R_pos = (R > 0).astype(np.int)
    #             IS = R_pos * mask
    #             union = (mask + R_pos) - IS
    #             IoU = IS.sum() / (union.sum() + 1e-8)
    #             print('IoU with thrshold 0 :', IoU)


    
    

def show_IoU_comp(img_list, label_oppo, hyper_list, hyper_path_list):
    for hyper in hyper_list:
        selected_hyper = fnmatch.filter(hyper_path_list, hyper)
        if len(selected_hyper) == 0:
            raise Exception('There is no such hyper_dir named {}'.format(hyper))
        elif len(selected_hyper) > 1:
            raise Exception('There are more than one {} path'.format(hyper))
        selected_hyper = selected_hyper[0]

        print(selected_hyper)

        with h5py.File(os.path.join(selected_hyper, 'mIOU_R_00.hdf5'), 'r') as f:
            print(list(f.keys()))
#             print('img_true_pos.shape :', f['0/true_pos/ori/img'].shape)
#             print('R_true_pos.shape :', f['0/true_pos/ori/R'].shape)
#             print('mask_true_pos.shape :', f['0/true_pos/ori/mask'].shape)
#             print()
            img_dict = collections.OrderedDict()
            img_dict['true_pos'] = f['0/true_pos/ori/img_idx'][()] if '0/true_pos/ori/img_idx' in f else np.array([[]])
            img_dict['false_pos'] = f['0/false_pos/ori/img_idx'][()] if '0/false_pos/ori/img_idx' in f else np.array([[]])
            img_dict['true_neg'] = f['0/true_neg/ori/img_idx'][()] if '0/true_neg/ori/img_idx' in f else np.array([[]])
            img_dict['false_neg'] = f['0/false_neg/ori/img_idx'][()] if '0/false_neg/ori/img_idx' in f else np.array([[]])
            
            for key in list(img_dict.keys()):
                print('{} : {}'.format(key, img_dict[key][:100]))
            print()
            
            print('################ Statistics ################')
            for trial_key in list(f.keys()):
                cof = []
                for cof_key in ['true_pos', 'false_pos', 'true_neg', 'false_neg']:
                    if '{}/{}/ori/img_idx'.format(trial_key, cof_key) in f:
                        cof.append(f['{}/{}/ori/img_idx'.format(trial_key, cof_key)].shape[0])
                    else:
                        cof.append(0)
                cof.append(sum(cof))
                print('{} | true_pos {}/{}, false_pos {}/{}, true_neg {}/{}, false_neg {}/{}, sum {}'.format( \
                trial_key, cof[0], round(cof[0]/cof[4], 2), \
                           cof[1], round(cof[1]/cof[4], 2), \
                           cof[2], round(cof[2]/cof[4], 2), \
                           cof[3], round(cof[3]/cof[4], 2), cof[4]))
            
            for img_idx in img_list:
                print('============================ {} ============================'.format(img_idx))
                for key in list(img_dict.keys()): # key = 'true_pos' or 'true_pos' ...
                    if img_idx in img_dict[key]: # img_idx = img number that I chose
                        idx = np.where(img_dict[key] == img_idx) # 몇번째에 원하는 img가 있는지 찾는다
                        img = f['0/{}/ori/img'.format(key)][list(idx[0])[0]][0] # 첫번째 슬라이드만 visualize
                        mask = f['0/{}/ori/mask'.format(key)][list(idx[0])[0]][0]
                        R_ori = f['0/{}/ori/R'.format(key)][list(idx[0])[0]][0]
                        R_ori = skimage.transform.resize(R_ori, (224, 224))
                        if label_oppo:
                            R_oppo = f['0/{}/oppo/R'.format(key)][list(idx[0])[0]][0]
                            R_oppo = skimage.transform.resize(R_oppo, (224, 224))


                        union, IS, IoU = None, None, None
                        if key == 'true_pos':
                            union = f['0/{}/ori/union'.format(key)][list(idx[0])[0]][0]
                            IS = f['0/{}/ori/IS'.format(key)][list(idx[0])[0]][0]
                            IoU = f['0/{}/ori/IoU'.format(key)][list(idx[0])[0]]

                            print('key :', key)
                            print('IoU :', IoU)
                            fig, ax = plt.subplots(nrows=1, ncols=6 if label_oppo else 5, figsize=(20, 20))
                            ax[0].imshow(img, cmap='gray')
                            ax[0].axis('off')
                            ax[1].imshow(mask, cmap='gray')
                            ax[1].axis('off')
                            seaborn.heatmap(R_ori, cmap='RdBu_r', center=0, ax=ax[2], square=True, cbar=False, cbar_kws={'shrink': .2})
                            ax[2].axis('off')
                            if label_oppo:
                                seaborn.heatmap(R_oppo, cmap='RdBu_r', center=0, ax=ax[3], square=True, cbar=False, cbar_kws={'shrink': .2})
                                ax[3].axis('off')
                            ax[4 if label_oppo else 3].imshow(union, cmap='gray')
                            ax[4 if label_oppo else 3].axis('off')
                            ax[5 if label_oppo else 4].imshow(IS, cmap='gray')
                            ax[5 if label_oppo else 4].axis('off')
                            plt.show()


                        else:
                            print('key :', key)
                            fig, ax = plt.subplots(nrows=1, ncols=4 if label_oppo else 3, figsize=(20, 20))
                            ax[0].imshow(img, cmap='gray')
                            ax[0].axis('off')
                            ax[1].imshow(mask, cmap='gray')
                            ax[1].axis('off')
                            seaborn.heatmap(R_ori, cmap='RdBu_r', center=0, ax=ax[2], square=True, cbar=False, cbar_kws={'shrink': .2})
                            ax[2].axis('off')
                            if label_oppo:
                                seaborn.heatmap(R_oppo, cmap='RdBu_r', center=0, ax=ax[3], square=True, cbar=False, cbar_kws={'shrink': .2})
                                ax[3].axis('off')
                            plt.show()




def show_IoU_comp_example(name_list, hyper_list, hyper_path_list):
    assert len(name_list) == len(hyper_list), 'Different length'
    img_dict = collections.OrderedDict()
    for num, hyper in enumerate(hyper_list):
        selected_hyper = fnmatch.filter(hyper_path_list, hyper)
        if len(selected_hyper) == 0:
            raise Exception('There is no such hyper_dir named {}'.format(hyper))
        elif len(selected_hyper) > 1:
            raise Exception('There are more than one {} path'.format(hyper))
        selected_hyper = selected_hyper[0]

        print(selected_hyper)

        with h5py.File(os.path.join(selected_hyper, 'mIOU_R_00.hdf5'), 'r') as f:
            # algorithm과 cross val num을 구분해서 img_dict에 true_pos에 해당하는 image 번호와 그 image의 IoU를 저장
            img_dict['{}'.format(name_list[num])] = collections.OrderedDict()
            for crs_val_key in list(f.keys()):
                img_dict['{}'.format(name_list[num])]['{}'.format(crs_val_key)] = collections.OrderedDict()
                img_dict['{}'.format(name_list[num])]['{}'.format(crs_val_key)]['img_idx'] = \
                                                            f['{}/true_pos/ori/img_idx'.format(crs_val_key)][()]
                img_dict['{}'.format(name_list[num])]['{}'.format(crs_val_key)]['IoU'] = \
                                                            f['{}/true_pos/ori/IoU'.format(crs_val_key)][()]

    print(list(img_dict.keys()))
    # 특정 cross val num에서
    for crs_val_key in list(img_dict[name_list[0]].keys()):
        # 모든 image num 다 긁어모음
        all_img_nums = np.array([])
        for algo_key in list(img_dict.keys()):
            all_img_nums = np.concatenate([all_img_nums, img_dict[algo_key][crs_val_key]['img_idx']], axis=0)
        
        # 모든 image num에서 중복된 개수가 len(name_list)되면 즉, 모든 알고리즘에서 모두 true_pos로 존재하는 img넘버를 가려냄
        unique, counts = np.unique(all_img_nums.astype(np.int), return_counts=True)
        img_num_IS = unique[counts==len(name_list)]
        print(unique.shape, img_num_IS.shape)
        
        
        RSR_LRP_win_count = 0
        RSR_LRP_GA_win_from_clsfr_count = 0
        RSR_LRP_GA_win_from_RSR_LRP_count = 0
        for num in range(img_num_IS.shape[0]):
            all_IoUs = []
            for algo_key in list(img_dict.keys()):
                idx = img_dict[algo_key][crs_val_key]['img_idx'] == img_num_IS[num]
                all_IoUs.append(img_dict[algo_key][crs_val_key]['IoU'][idx].item())
            
            if all_IoUs[1] > all_IoUs[0]:
                RSR_LRP_win_count += 1
            if all_IoUs[2] > all_IoUs[0]:
                RSR_LRP_GA_win_from_clsfr_count += 1
                if all_IoUs[2] > all_IoUs[1]:
                    if all_IoUs[1] > all_IoUs[0]:
                        print('img_num_IS[num] :', img_num_IS[num])
                        pass
            if all_IoUs[2] > all_IoUs[1]:
                RSR_LRP_GA_win_from_RSR_LRP_count += 1
                
        print('RSR-LRP from clsfr :', RSR_LRP_win_count, round(RSR_LRP_win_count/img_num_IS.shape[0], 3))
        print('RSR-LRP+GA from clsfr :', RSR_LRP_GA_win_from_clsfr_count, round(RSR_LRP_GA_win_from_clsfr_count/img_num_IS.shape[0], 3))
        print('RSR-LRP+GA from RSR-LRP :', RSR_LRP_GA_win_from_RSR_LRP_count, round(RSR_LRP_GA_win_from_RSR_LRP_count/img_num_IS.shape[0], 3))



        
def show_interp_epoch(num_of_imgs, hyper_list, hyper_path_list):
    for hyper in hyper_list:
        selected_hyper = fnmatch.filter(hyper_path_list, hyper)
        if len(selected_hyper) == 0:
            raise Exception('There is no such hyper_dir named {}'.format(hyper))
        elif len(selected_hyper) > 1:
            raise Exception('There are more than one {} path'.format(hyper))
        selected_hyper = selected_hyper[0]

        print(selected_hyper)
        
        print('copying...')
        shutil.copyfile(os.path.join(selected_hyper, 'test.hdf5'), os.path.join(selected_hyper, 'test_copy.hdf5'))
        
        with h5py.File(os.path.join(selected_hyper, 'test_copy.hdf5'), 'r') as f:
            print(list(f.keys()))
            
            for epoch in range(len(list(f.keys()))):
                for i in range(num_of_imgs):
                    print('epoch :', epoch)
                    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
                    ax[0].imshow(f['{}/saving_masked_img'.format(epoch)][i,0,:,:], cmap='gray')
                    ax[0].set_title('masked_img', fontsize=15)
                    ax[1].imshow(f['{}/sig_mask'.format(epoch)][i,0,:,:], cmap='gray')
                    ax[1].set_title('sig_mask', fontsize=15)
                    ax[2].imshow(f['{}/R'.format(epoch)][i,0,:,:], cmap='gray')
                    ax[2].set_title('R', fontsize=15)
                    plt.show()

                print('='*30)
                
            
            
            
            
            
#             with h5py.File('result00.hdf5', 'r') as f:
#     print(list(f.keys()))
# #     print(f['sig_mask'][()].shape)
    
#             for epoch in range(30):
#                 for i in [i for i in range(2)]:
#                     print('epoch :', epoch)
#         #             print('가려진 input')
#         #             plt.imshow(f['{}/saving_masked_img'.format(epoch)][i,0,:,:], cmap='gray')
#         #             plt.show()
#         #             print('mask')
#         #             plt.imshow(f['{}/sig_mask'.format(epoch)][i,0,:,:], cmap='gray')
#         #             plt.show()
#         #             print('R')
#         #             plt.imshow(f['{}/R'.format(epoch)][i,0,:,:], cmap='gray')
#         #             plt.show()

#                     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
#                     ax[0].imshow(f['{}/saving_masked_img'.format(epoch)][i,0,:,:], cmap='gray')
#                     ax[1].imshow(f['{}/sig_mask'.format(epoch)][i,0,:,:], cmap='gray')
#                     ax[2].imshow(f['{}/R'.format(epoch)][i,0,:,:], cmap='gray')
#                     plt.show()

#                 print('='*30)








    
    
            

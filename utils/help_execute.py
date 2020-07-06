import os
import h5py
import shutil
import fnmatch
import numpy as np
import collections


def make_execute_bash(ordereddict, filename, path_to_main, result_dir, test_model_path=None):
    total_line_num = 1
    for key in ordereddict:
        total_line_num *= len(ordereddict[key])

#     print(total_line_num)
    
#     line_list = ['echo "asdf" | sudo -S python3 main_tmp/main_{}.py'.format(result_dir[7:]) for i in range(total_line_num)]
    line_list = ['echo "asdf" | sudo -S python3 main.py' for i in range(total_line_num)]
    
#     fixed_string = ['result_dir={} &&\ndatetime=$(date +%Y%m%d%H%M%S%3N) &&\nresult_path=$result_dir/$datetime &&\necho \"asdf\" | sudo -S mkdir \"$result_dir/$datetime\" &&\necho \"asdf\" | sudo -S touch $result_dir/$datetime/setting.txt &&\necho \"asdf\" | sudo -S chmod a=rwx ./*\necho \"asdf\" | sudo -S chmod a=rwx $result_dir/$datetime/*'.format(result_dir)]
    
    fixed_string = ['result_dir={} &&\ndatetime=$(date +%Y%m%d%H%M%S%3N) &&\nresult_path=$result_dir/$datetime &&\necho \"asdf\" | sudo -S mkdir \"$result_dir/$datetime\" &&\necho \"asdf\" | sudo -S touch $result_dir/$datetime/setting.txt &&\necho \"asdf\" | sudo -S chmod a=rwx ./*\necho \"asdf\" | sudo -S chmod a=rwx $result_dir/$datetime/*\ncd {}\nresult_dir=../../../{}\nresult_path=$result_dir/$datetime'.format(result_dir, path_to_main, result_dir)]


#     for line in line_list:
#         print(line)
    
    # num_of_formal_hypos_combination 아래 loop에서 이전까지의 iter들에 나왔던 hypo들의 combination 수 만큼 
    # 이번 iter의 hypo의 각 element를 반복해야 하므로 이전까지의 iter들에 나왔던 hypo들의 combination 수를 담는 변수임.
    num_of_formal_hypos_combination = 1
    for key in ordereddict:
        hypo_value_list = ordereddict[key]
        hypo_value_elementwise_repeated_list = []
        for hypo in hypo_value_list:
            hypo_value_elementwise_repeated_list += [hypo for i in range(num_of_formal_hypos_combination)]
#         print('1. num_of_formal_hypos_combination :', num_of_formal_hypos_combination)
#         print('2. hypo_value_elementwise_repeated_list :', hypo_value_elementwise_repeated_list)

        num_of_formal_hypos_combination *= len(ordereddict[key])
#         print('3. num_of_formal_hypos_combination :', num_of_formal_hypos_combination)
        
        # num_of_latter_hypos_combination는 hypo_value_elementwise_repeated_list가 이후의 iter에 나오는 hypo들의 combination 수 만큼
        # elementwise하게 말고 전체 list가 반복되어야 하므로 이후에 나오는 hypo들의 combination 수를 담는 변수임.
        num_of_latter_hypos_combination = int(total_line_num/num_of_formal_hypos_combination)
        hypo_value_repetition_list = [str(i) for i in (hypo_value_elementwise_repeated_list*num_of_latter_hypos_combination)]
#         print('4. int(total_line_num/num_of_formal_hypos_combination) :', int(total_line_num/num_of_formal_hypos_combination))
#         print('5. hypo_value_repetition_list :', hypo_value_repetition_list, type(hypo_value_repetition_list))
#         line_list = [line+' --{} {}'.format(key, hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
    
        # key와 python main file argument의 변수이름이 다른 경우를 처리하기 위한 if절 
        if key == 'trial':
            line_list = [line+' -t {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'fold':
            line_list = [line+' -f {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'model':
            line_list = [line+' -a {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'custom_init':
            line_list = [line+' --custom-init {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'init_name':
            line_list = [line+' --init-name {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'epoch':
            line_list = [line+' --epochs {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'mask_path':
            line_list = [line+' --mask-path {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        elif key == 'batch_size':
            line_list = [line+' --batch-size {}'.format(hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
        else:
            line_list = [line+' --{} {}'.format(key, hypo_value_repetition_list[num]) for num, line in enumerate(line_list)]
    line_list = [line+' -rp $result_path --smooth_std 1.0 --smooth_num 50 --gamma 1.2 --img_name juna' for num, line in enumerate(line_list)]
    
    
    # hyperparameter directory를 생성할 때는 fold와 trial의 combination 수당 하나만 생성하므로 hyperparameter directory 생성을
    # 맡는 script부분은 이렇게 따로 만들어서 위에서 만든 line_list 사이사이에 끼워넣는다.
    hypo_record_skip_num = len(ordereddict['trial']) * len(ordereddict['fold'])
    line_list_for_record = ['echo \'asdf\' | sudo -S echo -e \"' for i in range(total_line_num//hypo_record_skip_num)]
    
    num_of_formal_hypos_combination = 1
    for key in ordereddict:
        hypo_value_list = ordereddict[key]
        hypo_value_elementwise_repeated_list = []
        for hypo in hypo_value_list:
            hypo_value_elementwise_repeated_list += [hypo for i in range(num_of_formal_hypos_combination)]
        
        if key in ['fold', 'trial']:
            pass
        else:
            num_of_formal_hypos_combination *= len(ordereddict[key])

        num_of_latter_hypos_combination = int(total_line_num/num_of_formal_hypos_combination)
        hypo_value_repetition_list = [str(i) for i in (hypo_value_elementwise_repeated_list*num_of_latter_hypos_combination)]
        
        
        if key == 'fold':
            pass
        elif key == 'trial':
            line_list_for_record = [line+'trial : {}\\n'.format(ordereddict[key][-1]+1) for num, line in enumerate(line_list_for_record)]
        else:
            line_list_for_record = [line+'{} : {}\\n'.format(key, hypo_value_repetition_list[num]) for num, line in enumerate(line_list_for_record)]
    additional_str = 'result_dir : $result_dir\\nhyper_dir : $datetime\" >> $result_dir/$datetime/setting.txt'
    line_list_for_record = [line+additional_str for num, line in enumerate(line_list_for_record)]
    
    if test_model_path == None:
        additional_str2 = ['echo "asdf" | sudo -S python3 ./$result_dir/crossval_plot-1909291726.py ./$result_dir\ncd ../../../ #']
    else:
        additional_str2 = []
    
    hypo_record_skip_num = len(ordereddict['trial']) * len(ordereddict['fold'])
    new_line_list = []
    for line_num in range(len(line_list)):
        if line_num % hypo_record_skip_num == 0:
            line_num_to_insert = line_num // hypo_record_skip_num
            new_line_list += [''] + \
                             fixed_string + \
                             [line_list_for_record[line_num_to_insert]] + \
                             [''] + \
                             line_list[line_num:line_num+hypo_record_skip_num] + \
                             additional_str2
                             
    
#     print('##########################################')
#     for line in new_line_list:
#         print(line)
#     print('##########################################')
    
    with open(filename, 'w') as f:
        for num, line in enumerate(new_line_list):
            if num == len(new_line_list)-1:
                f.write('{}'.format(line))
            elif line == '':
                f.write('{}\n'.format(line))
            else:
                f.write('{} &&\n'.format(line))

                
def get_things_to_do_after_experiment(filename, path):
    file_list = [file for file in os.listdir('../Things_to_do_after_experiment') if 'py' in file]
    file_list.sort()

    file_wanted = [file for file in file_list if file[:file.find('-')] == filename]
    file_wanted.sort()
    
    shutil.copyfile(os.path.join('../Things_to_do_after_experiment', file_wanted[-1]), os.path.join(path, file_wanted[-1]))

                
                
def get_setting(path):
    setting_dict = collections.OrderedDict()
    with open(os.path.join(path, 'setting.txt'), 'r') as f:
        key = None
        for line in f:
            if ':' in line:
                key = line[:line.find(':')-1]
                setting_dict[key] = [line[line.find(':')+2:-1]]
    return setting_dict     

def write_setting(setting_dict, path):
    with open(os.path.join(path, 'setting.txt'), 'w') as f:
        for key in list(setting_dict.keys()):
            f.write(key+' : '+setting_dict[key][0]+'\n')


def str_None(v):
    if v.lower() in ['None', 'none', 'no', 'Non']:
        return None
    else:
        return v

    
def get_hyperdir_by_setting(result_path, key, value):
    # 여기서 plot_hyper의 hyper는 relevance_map 폴더 안에서의 hyper_dir들을 의미함!!!
    plot_hyper = None

    result_dirs = [os.path.join(result_path, dir) for dir in fnmatch.filter(os.listdir(result_path), 're*')]
    for result_dir in result_dirs:
        hyper_dirs = [os.path.join(result_dir, dir) for dir in fnmatch.filter(os.listdir(result_dir), '20*')]
        for hyper_dir in hyper_dirs:
            setting_dict = get_setting(hyper_dir)
            if value in setting_dict[key][0]:
                plot_hyper = hyper_dir
    return plot_hyper


def ignore():
    return [
        '.*',
        'data',
        'git_repos',
        'before_*',
        'execute_*',
        'main_*',
        'project_tmp',
        'result_*',
        'move_result_files.ipynb',
        'plot_reset.ipynb',
        'table_for_project.ipynb',
        'test.ipynb',
        'relevance_execute_main.sh',
        '*.xlsx',
        'update_remote_server_result.py',
        'update_remote_server_result.sh',
        'project_cifar',
        'project_cifar_lrp',
        'testing_place',
        'Things_to_do_after_experiment',
    ]





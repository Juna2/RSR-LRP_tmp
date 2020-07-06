import sys
import torch
sys.path.append('../')
sys.path.append('../utils')
import time
import general
import datetime
import argparse
from models import custom_models

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
            
def str_None(v):
    if v.lower() in ['None', 'none', 'no', 'Non']:
        return None
    else:
        return v

def get_args():    
    parser = argparse.ArgumentParser(description='LRP')
    
    parser.add_argument('-t', '--trial', type=int, default=0,
                        help='number of different weight init trial.')
    parser.add_argument('-f', '--folder', type=int, default=0,
                        help='number of folder to use.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18') 
    parser.add_argument('--arch_for_test', default='None', type=str_None, metavar='ARCH')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--interpreter', type=str, default='lrp',
                        help='if it is lrp, lrp will be used as a interpreter. If not, gradcam will be used')
    parser.add_argument('--loss_type', type=str, default='mask_LRP_seg',
                        help='absR or uniformR or corner or frame or sparseR') ####
    parser.add_argument('--mask_system', type=str, metavar='PATH', default="mask_with_seg",
                        help='choose a mask system')
    parser.add_argument('--mask_data_size', type=int, default=56,
                        help='set the size of mask coming out from dataloader')
    parser.add_argument('--lambda_for_final', type=float, default=10,
                        help='if args. no_lambda_for_each_layer is True, regulizer rate for total loss. this is multiplied with lrp_loss')
    parser.add_argument('--lrp_target_layer', type=str_None, default='5',
                        help='label of mixed data 2')
    parser.add_argument('--R_process', type=str_None, default='max_norm',
                        help='How to process R after getting R at target layer')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='batch_size')
    parser.add_argument('--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--dataset_class', type=str, metavar='PATH', default="MSDDataset",
                        help='Determine dataset class')
    parser.add_argument("--small_dataset", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use small dataset for test")
    parser.add_argument('--data_path', metavar='DIR', default='./data/liver/2.size_224',
                    help='path to dataset')
    parser.add_argument('--mask-path', default=None, type=str,
                        help='path to mask file.')
    parser.add_argument("--custom-init", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use customized weight initialization")
    parser.add_argument('--init-name', type=str_None, metavar='PATH', default='false',
                        help='name of the weight customizing function')
    parser.add_argument("--save_relevance", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="save lrp results")  
    parser.add_argument('--input_channel_num', type=int, default=3,
                        help='input_channel_num') 
    parser.add_argument("--pretrained", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="use pretrained weight initialization")
    parser.add_argument('--threshold', type=float, help='IoU threshold.')
    parser.add_argument("--check_activ_output_shape", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="check_activ_output_shape") 
    parser.add_argument("--auc_as_acc", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="save auc instead of accuracy") 
    parser.add_argument('--metrics', type=str, default='IoU',
                        help='all measurements to do after training.')
    parser.add_argument("--data_aug", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Do data augmentation or not")
    parser.add_argument('--train_func', type=str, metavar='PATH', default="train",
                        help='Determine train function')
    parser.add_argument('--gamma', default=1.2, type=float, 
                        help='learning rate for grad alignment update')
    parser.add_argument('--loss_filter', type=str, default='pos',
                        help='From which data you want to calculate loss(lrp_loss). ex) true_pos, pos') 
    parser.add_argument("--label_oppo", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Get interpretation from opposite label")
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Ratio of how much train data want to use') 
    parser.add_argument('--mask_ratio', type=float, default=1.0,
                        help='Ratio of how much train data want to use') 
    parser.add_argument('-rp', '--result_path', type=str,
                        help='path to save the result.')
    parser.add_argument('--smooth_std', type=float, default=1.0,
                        help='gamma for fucntion heatmap() from render.py ')
    parser.add_argument('--smooth_num', type=int, default=50,
                        help='T.T')
    parser.add_argument('--img_name', type=str, default='juna',
                        help='img file naming (default: results )')
    parser.add_argument('--input_depth', type=int, default=3,
                        help='set input depth')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--interp_monitor', type=str2bool, default=False,
                        help='save interp every epoch')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--loader_IoU', type=str, default='val',
                        help='data loader used for IoU') 
    parser.add_argument('--IoU_compact_save', type=str2bool, default=False,
                        help='save only IoU when calculation IoU')
    parser.add_argument("--label_oppo_status", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="This is set by args.label_oppo_status so do not touch it")
    parser.add_argument('--frame_mask', type=str2bool, default=False,
                        help='use frame mask instead of segmentation label')
    parser.add_argument('--num_skip_cls_loss', type=int, default=1,
                        help='num_classes')
    
    
    
    
    
    
    
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--pretrained_path_num', type=int, default=None,
                        help='number of path of pretrained model')
    parser.add_argument('--freeze_leq', type=int, default=-1,
                        help='freeze layers that less or equal to given number')
    parser.add_argument('--vgg_bn', action='store_true', default=False,
                        help='if you want to get from_acc when targeted ')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='whether save model or not')
    parser.add_argument('--saved_model', type=str, default=None, 
                        help='The model you want to load from run_lrp/trained_model/.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--pretrained-model-path', default='None', type=str_None, metavar='PATH',
                        help='use pre-trained model & the path of the model')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num_classes')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--test_model_path', default='None', type=str_None, metavar='PATH',
                        help='use pre-trained model & the path of the model')
    
    
    #### 2. Interpreter ####
    
    parser.add_argument('--r-method', type=str, default='composite',
                        help='relevance methods: simple/epsilon/w^2/alphabeta/composite/new_composite/flrp_acw/flrp_aw')
    parser.add_argument('--no-relevance', action='store_true', default=False,
                        help='do not compute relevance')
    parser.add_argument('--new_lrp_proportion', type=float, default=1.,
                        help='proportional constant of Fisher ')
    parser.add_argument('--new_lrp_epsilon', type=float, default=1.,
                        help='epsilon constant of Fisher ')
    parser.add_argument('--new_lrp_device', type=int, default=0,
                        help='main device to run new_lrp in convolution.py')
    
    #### 3. Loss ####
    parser.add_argument('--train_by', type=str, default='total_loss',
                        help='quanti_type in def quanti_metric from sequential.py.. sign or rank or frame')
    parser.add_argument('--clip', type=float, default=10,
                        help='how clipping')
    parser.add_argument('--entropy_type', type=str, default='abs',
                        help='all_positive or abs, softmax')
    parser.add_argument('--uniformR_type', type=str, default='entropy',
                        help='variance or else')
    parser.add_argument('--sparseR_type', type=str, default='entropy',
                        help='variance or L1 or entropy')
    
    #### 4. Visualization ####
    
    parser.add_argument('--beta', type=float, default=0,
                        help='beta for alpha beta lrp')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='beta for alpha beta lrp')
    parser.add_argument('--random_seed', type=int, default=20,
                        help='random seed for visualization')
    
    parser.add_argument('--num_visualize_plot', type=int, default=100,
                        help='number of img to see') 
    
    #### 5. Active fooling ####
    parser.add_argument('--heatmap_label', type=int, default=0,
                        help='the class label that you want to get heatmap')
    parser.add_argument('--class_from', type=int, default=555,
                        help='label of mixed data 1')
    parser.add_argument('--class_to', type=int, default=366,
                        help='label of mixed data 2')
    
    #### 6. Evaluation ####
    parser.add_argument('--imagenet_accuracy', action='store_true', default=False, #0.5,
                        help='evaluate the accuracy of validation dataset of Imagenet.')
    parser.add_argument('--from_accuracy', action='store_true', default=False,
                        help='if you want to get from_acc when targeted ')
    parser.add_argument('--to_accuracy', action='store_true', default=False,
                        help='if you want to get to_acc when targeted ')
    parser.add_argument('--quanti_type', type=str, default='sign',
                        help='quanti_type in def quanti_metric from sequential.py.. sign or rank or frame')
    parser.add_argument('--get_correlation', action='store_true', default=False,
                        help='if you want to get from_acc when targeted ')
    
    
    #### 7. Fisher ####
    parser.add_argument('--fisher_iter', type=int, default=0,
                        help='The number of iterations of saved model')
    parser.add_argument('--fisher_interpreter', type=str, default='ERROR',
                        help='grad_cam/lrp/lrp34/simple/simple34')
    parser.add_argument('--fisher_save_dir', type=str, default='ERROR',
                        help='save directory for fisher experiment')
    parser.add_argument('--fisher_c', type=str, default='ERROR',
                        help='fc/gc')
    parser.add_argument('--fisher_r', type=str, default='ERROR',
                        help='gr/ge')
    
    
    #### 8. FGSM ####
    parser.add_argument('--FGSM_type', type=str, default='nontarget',
                        help='nontarget or target')
    parser.add_argument('--FGSM_loss_type', type=str, default='topk',
                        help='topk, return_p1, return_p2, mass_center')
    parser.add_argument('--fgsm_iteration', type=int, default=20,
                        help='how many time you want to iterate to get perturbed image')
    
    
    #### 9. Save Result ####
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--log-dir', default='./results_ImageNet/',
                        help='directory to save logs (default: ./results_MNIST)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
    parser.add_argument('--img_dir', default='../img_results/', 
                        help='img (default: )')
    parser.add_argument('--port', type=int, default=8085,
                        help='port to run the server on (default: 8085)')
    parser.add_argument('--no-save-model', action='store_false', default=True,
                        help='do not save model')
    parser.add_argument('--reload_model', action='store_true', default=False,
                        help='Restore the trained model')
    parser.add_argument('--pretrained_model_name', type=str, default='None',
                        help='name of pretrained model to check')
    parser.add_argument('--r-filename', default='./results_MNIST/train/relevance_file',
                        help='file name for relevance')
    
    #### 10. Undetermined ####
    parser.add_argument('--whichScore',  type=int, default=None, #8,
                        help='score that you want to see/ can be 0~9 in MNIST')
    parser.add_argument('--lambda_dic', nargs='+', type=int, default= [1,0,1,0,1,0,1], #0.5,
                        help='regulizer rate of each layer for total loss. this is multiplied with each layer lrp_loss. should be equal to the number of layers. [layer1, layer2, layer3, ...]')
    parser.add_argument('--no_lambda_for_each_layer', action='store_true', default=True, #0.5,
                        help='wheather to use regulizer rate of each layer for total loss. this is multiplied with each layer lrp_loss')
    parser.add_argument('--lambda_mode', type=str, default='swith_on',
                        help='lambda_mode: no_change/swith_on/increasing') 
    parser.add_argument('--lambda_switch_ep', type=int, default=1,
                        help='lambda_switch_ep (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=False, 
                        help='disables visdom visualization')
    parser.add_argument('--no-bias', action='store_true', default=True,
                        help='do not consider bias relevance')
    parser.add_argument('--eval-period', type=int, default=5,
                        help='validation set evaluation period (default: 5)')
    parser.add_argument('--num_eval', type=int, default=5,
                        help='validation set number of evaluation (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of processes for dataloader (default: 4)')
    parser.add_argument('--test-every', type=int, default=1,
                        help='run test every test_every. (default: 1)')
    parser.add_argument('--lambda0_point', type=int, default=1000000,
                        help='lambda0point')
    parser.add_argument('--after_lr', type=float, default=0.001,
                        help='after_lr')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer (default: Adam)')
    parser.add_argument('--dropout', type=float, default=1,
                        help='Keep probability for training dropout. (default: 1)')
    parser.add_argument('--data-normalization', default="False",
                        help='data_normalization or not')
    parser.add_argument('--salt-and-pepper', action='store_true',default="False",
                        help='salt-and-pepper or not')
    parser.add_argument('--flip_ratio', type=float, default=0.1,
                        help='salt-and-pepper flip ratio')
    parser.add_argument('--max_ratio', type=float, default=0.5,
                        help='salt-and-pepper max ratio')
    parser.add_argument('--img_size', type=int, default=784,
                        help='size of img. this is for normalizing lrp loss- defalt:28*28(MNIST)')
    parser.add_argument('--visualize_kind', type=str, default='graph',
                        help='graph  or  img')
    parser.add_argument('--yes_clip', action='store_true', default=False,
                        help='gradient clipping or not')
    
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    
    
    
    

    return args

import os
import sys
import time
import h5py
import torch
import skimage
import general
import register
import numpy as np
import sklearn.metrics as metrics

from tqdm import tqdm
from torch.utils.data import Dataset


def get_masked_img(images, R):
    w = 20
    sigma = 0.5
    R_abs = torch.nn.functional.interpolate(torch.abs(R), (224))
    mask = 1 - torch.sigmoid(w*(R_abs-sigma))
    masked_img = images * mask
    masked_img_new_graph = masked_img.detach().requires_grad_()
    return masked_img_new_graph, masked_img, mask


def get_masked_img_no_grad_skip(images, R):
    w = 20
    sigma = 0.5
    R_abs = torch.nn.functional.interpolate(torch.abs(R), (224))
    mask = 1 - torch.sigmoid(w*(R_abs-sigma))
    masked_img = images * mask
    return masked_img, mask

# def get_mask(gcam, sigma=.5, w=8):
#     gcam = (gcam - F.min(gcam).data)/(F.max(gcam) - F.min(gcam)).data
#     mask = F.squeeze(F.sigmoid(w * (gcam - sigma)))
#     return mask


def thr_max_norm(R):
    R = torch.nn.functional.threshold(R, threshold=0, value=0)
    R_max,_ = R.max(2, keepdim=True)
    R_max,_ = R_max.max(3, keepdim=True)
    R = R / (R_max + 1e-8)
    return R

def max_norm(R):
    R_abs = torch.abs(R)
    R_max,_ = R_abs.max(2, keepdim=True)
    R_max,_ = R_max.max(3, keepdim=True)
    R = R / (R_max + 1e-8)
    return R

@register.IoU_arg_setting
def get_mIoU(data_loader, model, criterion, args): 

    total_img = []
    total_R = []
    total_union = []
    total_IS = []
    total_IoU = []
    total_mask = []
    total_output = []
    total_label = []
    
    model.eval()
    
    # mask size를 input size와 같게 키워줘야 함.
    ori_mask_size = args.mask_data_size
    data_loader.dataset.mask_size = None # None으로 설정하면 mask가 저장되어 있을 때와 동일한 size로 나옴
    
    with torch.no_grad():
        
        # label이 가리키는 score의 반대 score에서도 lrp를 내리기 위한 장치
        status_list = None
        if args.label_oppo:
            status_list = [False, True]
        else:
            status_list = [False]
            
        for status in status_list:
            # args.label_oppo_status가 True면 label의 반대에서 lrp를 내린다.
            args.label_oppo_status = status
            inter_from = None
            if status:
                inter_from = 'oppo'
            else:
                inter_from = 'ori'
                
            img_cul_num = 0 # val img들에 번호를 매겨 다른 setting의 R들과 비교하기 편리하게 하기 위함
            for (data, label, use_mask) in tqdm(data_loader):
                img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
                label = label.cuda(args.gpu)

                img_idx = np.arange(label.shape[0]) + img_cul_num

                output, lrp_loss, R_ori = model.forward(img, label, mask, use_mask, args) # R, R_ori ####

                pred = torch.argmax(output, dim=1)
                true_false = (pred == label).type(torch.cuda.LongTensor)

                iou_path = os.path.join(args.result_path, 'mIOU_R_{:02d}.hdf5'.format(args.folder))

                # R의 양수만 남기고 normalize한 것으로 IoU구함
                R = thr_max_norm(R_ori) # IoU 계산할 때 사용
                R_vis = max_norm(R_ori) # Visualize할 때 사용

                # label은 tumor이고 pred도 tumor로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함)
                true_pos = (true_false * pred).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_true_pos = R[true_pos].cpu().numpy()
                img_idx_true_pos = img_idx[true_pos]
                if R_true_pos.shape[0] != 0:
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, 
                                          '{}/true_pos/{}/R'.format(args.trial, inter_from), 
                                          R_vis[true_pos].cpu().numpy()) ####

                    R_true_pos = skimage.transform.resize(R_true_pos, \
                                         (R_true_pos.shape[0], R_true_pos.shape[1], args.mask_data_size, args.mask_data_size))

                    mask_true_pos = mask.cpu().numpy()[true_pos]
                    mask_true_pos = (mask_true_pos > 0).astype(np.int)
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, '{}/true_pos/{}/mask'.format(args.trial, inter_from), mask_true_pos)

                    # R의 상위 (args.threshold)%에 해당하는 value가 무엇인지 알아내서 bound에 담기
    #                 R_true_pos_tmp = R_true_pos.reshape(R_true_pos.shape[0], -1)
    #                 bound = int(R_true_pos_tmp.shape[1] * args.threshold)
    #                 bound = -np.sort(-R_true_pos_tmp, axis=1)[:, bound] # 여기서 hist 그려서 상위 #%에 맞게 bound 잘 설정되는지 확인
    #                 bound = bound[..., np.newaxis, np.newaxis, np.newaxis]

                    R_true_pos_pos = (R_true_pos > args.threshold).astype(np.int)
                    IS = R_true_pos_pos * mask_true_pos
                    union = (mask_true_pos + R_true_pos_pos) - IS
                    IoU = (np.sum(IS.reshape(IS.shape[0], -1), axis=1) / (np.sum(union.reshape(union.shape[0], -1), axis=1) + 1e-8)) * 100

                    img_true_pos = img.cpu().numpy()[true_pos]
                    output_true_pos = output.cpu().numpy()[true_pos]
                    label_true_pos = label.cpu().numpy()[true_pos]
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, '{}/true_pos/{}/img'.format(args.trial, inter_from), img_true_pos)
                        general.save_hdf5(iou_path, '{}/true_pos/{}/union'.format(args.trial, inter_from), union)
                        general.save_hdf5(iou_path, '{}/true_pos/{}/IS'.format(args.trial, inter_from), IS)
                        general.save_hdf5(iou_path, '{}/true_pos/{}/img_idx'.format(args.trial, inter_from), img_idx_true_pos)
                        general.save_hdf5(iou_path, '{}/true_pos/{}/output'.format(args.trial, inter_from), output_true_pos)
                        general.save_hdf5(iou_path, '{}/true_pos/{}/label'.format(args.trial, inter_from), label_true_pos)
                    general.save_hdf5(iou_path, '{}/true_pos/{}/IoU'.format(args.trial, inter_from), IoU)


                # label은 liver인데 pred는 tumor로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함)
                false_pos = ((1 - true_false) * pred).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_false_pos = R[false_pos].cpu().numpy()
                img_idx_false_pos = img_idx[false_pos]
                if R_false_pos.shape[0] != 0:
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, 
                                          '{}/false_pos/{}/R'.format(args.trial, inter_from), 
                                          R_vis[false_pos].cpu().numpy()) ####

                    img_false_pos = img.cpu().numpy()[false_pos]
                    mask_false_pos = mask.cpu().numpy()[false_pos]
                    output_false_pos = output.cpu().numpy()[false_pos]
                    label_false_pos = label.cpu().numpy()[false_pos]
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, '{}/false_pos/{}/img'.format(args.trial, inter_from), img_false_pos)
                        general.save_hdf5(iou_path, '{}/false_pos/{}/mask'.format(args.trial, inter_from), mask_false_pos)
                        general.save_hdf5(iou_path, '{}/false_pos/{}/img_idx'.format(args.trial, inter_from), img_idx_false_pos)
                        general.save_hdf5(iou_path, '{}/false_pos/{}/output'.format(args.trial, inter_from), output_false_pos)
                        general.save_hdf5(iou_path, '{}/false_pos/{}/label'.format(args.trial, inter_from), label_false_pos)

                # label은 liver인데 pred도 liver로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함)
                true_neg = (true_false * (1 - pred)).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_true_neg = R[true_neg].cpu().numpy()
                img_idx_true_neg = img_idx[true_neg]
                if R_true_neg.shape[0] != 0:
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, 
                                          '{}/true_neg/{}/R'.format(args.trial, inter_from), 
                                          R_vis[true_neg].cpu().numpy()) ####

                    img_true_neg = img.cpu().numpy()[true_neg]
                    mask_true_neg = mask.cpu().numpy()[true_neg]
                    output_true_neg = output.cpu().numpy()[true_neg]
                    label_true_neg = label.cpu().numpy()[true_neg]
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, '{}/true_neg/{}/img'.format(args.trial, inter_from), img_true_neg)
                        general.save_hdf5(iou_path, '{}/true_neg/{}/mask'.format(args.trial, inter_from), mask_true_neg)
                        general.save_hdf5(iou_path, '{}/true_neg/{}/img_idx'.format(args.trial, inter_from), img_idx_true_neg)
                        general.save_hdf5(iou_path, '{}/true_neg/{}/output'.format(args.trial, inter_from), output_true_neg)
                        general.save_hdf5(iou_path, '{}/true_neg/{}/label'.format(args.trial, inter_from), label_true_neg)

                # label은 tumor인데 pred는 liver로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함) 
                false_neg = ((1 - true_false) * (1 - pred)).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_false_neg = R[false_neg].cpu().numpy()
                img_idx_false_neg = img_idx[false_neg]
                if R_false_neg.shape[0] != 0:
                    general.save_hdf5(iou_path, '{}/false_neg/{}/R'.format(args.trial, inter_from), R_vis[false_neg].cpu().numpy()) ####

                    img_false_neg = img.cpu().numpy()[false_neg]
                    mask_false_neg = mask.cpu().numpy()[false_neg]
                    output_false_neg = output.cpu().numpy()[false_neg]
                    label_false_neg = label.cpu().numpy()[false_neg]
                    if not args.IoU_compact_save:
                        general.save_hdf5(iou_path, '{}/false_neg/{}/img'.format(args.trial, inter_from), img_false_neg)
                        general.save_hdf5(iou_path, '{}/false_neg/{}/mask'.format(args.trial, inter_from), mask_false_neg)
                        general.save_hdf5(iou_path, '{}/false_neg/{}/img_idx'.format(args.trial, inter_from), img_idx_false_neg)
                        general.save_hdf5(iou_path, '{}/false_neg/{}/output'.format(args.trial, inter_from), output_false_neg)
                        general.save_hdf5(iou_path, '{}/false_neg/{}/label'.format(args.trial, inter_from), label_false_neg)



                img_cul_num += label.shape[0]
                ############################################################################
            
        print('mIOU_R_{:02d}.hdf5 saved'.format(args.folder))
    
    data_loader.dataset.mask_size = ori_mask_size


# def get_mIoU(val_loader, model, criterion, args):
#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         end = time.time()
#         #################
#         total_img = []
#         total_R = []
#         total_union = []
#         total_IS = []
#         total_IoU = []
#         total_mask = []
#         total_output = []
#         total_label = []
#         #################
#         for i, bundle in enumerate(tqdm(val_loader)):           
#             args.interpreter = 'lrp'
#             args.loss_type = 'None'
#             args.lrp_target_layer = None
            
#             output, lrp_loss, R = model.forward(images, label, args, mask_modi) #### mask_modi was originally mask
#             pred = torch.argmax(output, dim=1)
#             true_false = (pred == label).type(torch.cuda.LongTensor)
            
#             iou_path = os.path.join(args.result_path, 'mIOU_R_{:02d}.hdf5'.format(args.folder))
            
#             # label은 tumor이고 pred도 tumor로 한 경우의 interpretation을 골라내기 위함
#             # (interpretation은 label이 가리키는 score에 대해서 구함)
#             true_pos = (true_false * pred).type(torch.BoolTensor)
#             R = R / torch.abs(R).max()
#             R_true_pos = R[true_pos].cpu().numpy()
#             if R_true_pos.shape[0] != 0:
#                 general.save_hdf5(iou_path, 'R_true_pos', R_true_pos)
                
#                 mask_modi_true_pos = mask_modi.cpu().numpy()[true_pos]
#                 general.save_hdf5(iou_path, 'mask_modi_true_pos', mask_modi_true_pos)
            
#                 R_true_pos_pos = (R_true_pos > args.threshold).astype(np.int)
#                 IS = R_true_pos_pos * mask_modi_true_pos
#                 union = (mask_modi_true_pos + R_true_pos_pos) - IS
#                 IoU = (np.sum(IS.reshape(IS.shape[0], -1), axis=1) / (np.sum(union.reshape(union.shape[0], -1), axis=1) + 1e-8)) * 100
                
#                 img_true_pos = images.cpu().numpy()[true_pos]
#                 general.save_hdf5(iou_path, 'img_true_pos', img_true_pos)
#                 general.save_hdf5(iou_path, 'union', union)
#                 general.save_hdf5(iou_path, 'IS', IS)
#                 general.save_hdf5(iou_path, 'IoU', IoU)

            
#             # label은 liver인데 pred는 tumor로 한 경우의 interpretation을 골라내기 위함
#             # (interpretation은 label이 가리키는 score에 대해서 구함)
#             false_pos = ((1 - true_false) * pred).type(torch.BoolTensor)
#             R_false_pos = R[false_pos].cpu().numpy()
#             if R_false_pos.shape[0] != 0:
#                 general.save_hdf5(iou_path, 'R_false_pos', R_false_pos)
                
#                 img_false_pos = images.cpu().numpy()[false_pos]
#                 general.save_hdf5(iou_path, 'img_false_pos', img_false_pos)
#                 mask_modi_false_pos = mask_modi.cpu().numpy()[false_pos]
#                 general.save_hdf5(iou_path, 'mask_modi_false_pos', mask_modi_false_pos)
            
#             # label은 liver인데 pred도 liver로 한 경우의 interpretation을 골라내기 위함
#             # (interpretation은 label이 가리키는 score에 대해서 구함)
#             true_neg = (true_false * (1 - pred)).type(torch.BoolTensor)
#             R_true_neg = R[true_neg].cpu().numpy()
#             if R_true_neg.shape[0] != 0:
#                 general.save_hdf5(iou_path, 'R_true_neg', R_true_neg)
                
#                 img_true_neg = images.cpu().numpy()[true_neg]
#                 general.save_hdf5(iou_path, 'img_true_neg', img_true_neg)
#                 mask_modi_true_neg = mask_modi.cpu().numpy()[true_neg]
#                 general.save_hdf5(iou_path, 'mask_modi_true_neg', mask_modi_true_neg)
            
#             # label은 tumor인데 pred는 liver로 한 경우의 interpretation을 골라내기 위함
#             # (interpretation은 label이 가리키는 score에 대해서 구함)
#             false_neg = ((1 - true_false) * (1 - pred)).type(torch.BoolTensor)
#             R_false_neg = R[false_neg].cpu().numpy()
#             if R_false_neg.shape[0] != 0:
#                 general.save_hdf5(iou_path, 'R_false_neg', R_false_neg)
                
#                 img_false_neg = images.cpu().numpy()[false_neg]
#                 general.save_hdf5(iou_path, 'img_false_neg', img_false_neg)
#                 mask_modi_false_neg = mask_modi.cpu().numpy()[false_neg]
#                 general.save_hdf5(iou_path, 'mask_modi_false_neg', mask_modi_false_neg)
            
            
# #             total_img.append(images.cpu().numpy())
# #             total_R.append(R)
# #             total_union.append(union)
# #             total_IS.append(IS)
# #             total_IoU.append(IoU)
# #             total_mask.append(mask_modi)
# #             total_output.append(output.cpu().numpy())
# #             total_label.append(label.cpu().numpy())
            
            
#             general.save_hdf5(iou_path, 'output', output.cpu().numpy())
#             general.save_hdf5(iou_path, 'label', label.cpu().numpy())
#             ############################################################################

#             # measure elapsed time
#             end = time.time()

#         print('mIOU_R_{:02d}.hdf5 saved'.format(args.folder))
        
        
def get_AOPC(val_loader, model, criterion, mask, args):

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        #################
        total_R = []
        total_output = []
        total_label = []
        
        perturbed_result_path = os.path.join(args.result_path, 'perturbed_result.hdf5')
        #################
        for batch_num, bundle in enumerate(tqdm(val_loader)):
            ########################################
            if len(bundle) == 2:
                images, label = bundle
            elif len(bundle) == 3:
                images, label, mask = bundle

            images = images.type(torch.FloatTensor) 
            ########################################
            if args.arch[:5] == 'conv3':
                images = images.reshape(images.shape[0], 1, *images.shape[1:])
            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
                
            #################################################################################################
            if args.mask_system == 'one_mask_per_class':
                mask_modi = torch.from_numpy(mask[label.cpu().numpy()]).type(torch.cuda.FloatTensor)
            elif args.mask_system in ['mask_with_seg', 'only_one_mask']:
                mask_modi = mask.type(torch.cuda.FloatTensor)
            else:
                mask_modi = None
                
            if list(images.shape)[1:] == [1, 28, 28] and str(model[0])[:4] != 'Conv':
                images = images.reshape(images.shape[0], -1)
                mask_modi = mask_modi.reshape(mask_modi.shape[0], -1)
            ##################################################################################################
            
            ###########################################################################################################
            #######   Caution : Unlike normal AOPC. The dataset used here is ct image so its depth is not RGB  ########
            #######   so pixels which have the same depth should not be perturbed simultaneously.              ########
            ###########################################################################################################
            output, _, R = model.forward(images, label, args, mask=mask_modi) #### mask_modi was originally mask
            
            # interpretation, liver를 기준으로 perturb할 때, tumor의 pixel 개수만큼만 perturb하기 위해
            perturb_limit = get_perturb_limit(mask_modi, 67)
            
            ##############################################################################################
            ###############  perturb하는 영역을 제한하는 new_mask와 perturb의 순서를 정하는 R을 정의  ###############
            ##############################################################################################
            new_mask = None
            if args.perturb_reference == 'mask':
                new_mask = mask_modi.cpu().numpy()
                original_size = new_mask.shape[-1]
                
                # mask가 나타내는 tumor의 영역보다 같은 모양의 조금 더 넓은 영역을 가진 mask로 수정
                new_mask_resized = general.resize(new_mask, (67, 67))
                new_mask = general.resize(new_mask_resized, (original_size, original_size))
                new_mask = new_mask > 0
                
                new_mask = torch.from_numpy(new_mask).type(torch.cuda.FloatTensor)
                R = torch.from_numpy(np.random.rand(*R.shape)).type(torch.cuda.FloatTensor) * new_mask
            elif args.perturb_reference == 'random':
                # liver에 해당하는 pixel을 segment하는 mask를 만든다. 
                # 그리고 간에 해당하는 pixel만을 random으로 지우기 위한 R을 만든다.
                new_mask = images > -0.3214
#                 new_mask = new_mask.cpu().numpy()
            
                new_mask = new_mask.type(torch.cuda.FloatTensor)
                R = torch.from_numpy(np.random.rand(*R.shape)).type(torch.cuda.FloatTensor) * new_mask
            ###############################################################################################
            
            ###########################################################################
            ###############  label과 pred가 모두 tumor인 것들로만 AOPC 진행  ################
            ###########################################################################
            pred = torch.argmax(output, dim=1)
            true_false = (pred == label).type(torch.cuda.LongTensor)
            true_pos = (true_false * pred).type(torch.BoolTensor)
            
            if torch.sum(true_pos.type(torch.IntTensor)) == 0:
                continue
                
            R = R / torch.abs(R).max()
            
            images = images[true_pos]
            R = R[true_pos]
            perturb_limit = perturb_limit[true_pos]
            output = output[true_pos] 
            label = label[true_pos]
            ###########################################################################
            
            # 다 저장하기에는 용량이 많아서 [[0]]로 배치 당 첫번째에 있는 데이터 한개만 저장 ex) images[[0]]
            if batch_num < 50: # batch_num < 50일 때까지만 저장
                general.save_hdf5(perturbed_result_path, 'images', images[[0]].cpu().numpy())
                general.save_hdf5(perturbed_result_path, 'image_00', images[[0]].cpu().numpy())
                general.save_hdf5(perturbed_result_path, 'R', R[[0]].cpu().numpy())
                general.save_hdf5(perturbed_result_path, 'label', label.cpu().numpy())
                general.save_hdf5(perturbed_result_path, 'output', output[[0]].cpu().numpy())
                general.save_hdf5(perturbed_result_path, 'output_00', output.cpu().numpy()) # perturb를 전혀 하지 않았을때의 값부터 기록
                
                # 뒤에 나오는 perturb 중간에 나오는 output은 aopc계산을 위해 모두 저장해야하지만 
                # 각 image(batch당 한개만 저장)에 해당하는 output을 볼 때는 mask가 필요하므로 output_mask로 따로 저장
                output_shape = output.cpu().numpy().shape
                output_mask = np.zeros(output_shape[0])
                output_mask[0] += 1
                general.save_hdf5(perturbed_result_path, 'output_mask', output_mask.astype(np.bool))
                
            # R값이 큰 것부터 perterb하기 위해 -R 사용
            R = R.reshape(R.shape[0], -1)
            R_argsort = torch.argsort(-R, dim=1)
            row = torch.arange(images.shape[0]).unsqueeze(1)
                
            depth = images.shape[1] # 나중에 다시 reshape할 때 필요
            
            if len(images.shape) > 2:
                images = images.reshape(images.shape[0], -1)
            
            # 이 부분에서는 mask 이외의 부분이 perturb되는 것을 막기 위해 mask의 0에 해당하는 부분의 image를 미리 저장해놓는다.
            images_with_zero_on_mask = None
            if args.perturb_reference == 'mask' or args.perturb_reference == 'random':
                new_mask = new_mask[true_pos]
                
                if batch_num < 50:
                    general.save_hdf5(perturbed_result_path, 'mask', new_mask[[0]].cpu().numpy())
                    
                new_mask = new_mask.reshape(new_mask.shape[0], -1)
                images_with_zero_on_mask = images * (1-new_mask)
            
            # perturb는 n번 만큼 하지만 perturb를 전혀 하지 않았을때의 값부터 기록되므로 기록이 되는 output은 총 n+1개가 저장됨.
            for perturb_num in range(1, args.total_perturb_num+1):
                
                if len(images.shape) > 2:
                    images = images.reshape(images.shape[0], -1) ####
                    
                ################################################################################################
                ##########################  input pixel에 집어넣을 perturb matrix 만들기  ##########################
                ################################################################################################
                image_num = images.shape[0]
                perturb_rate = args.perturb_rate
                perturb_matrix = torch.from_numpy(np.random.normal(0, 1, image_num*perturb_rate).reshape(image_num, perturb_rate))
                
                if 'cifar' in args.data_path:
                    perturb_matrix = perturb_matrix.type(torch.IntTensor)
                    
                perturb_matrix = perturb_matrix.type(torch.FloatTensor)
                
                if args.gpu is not None:
                    perturb_matrix = perturb_matrix.cuda(args.gpu, non_blocking=True)
                ################################################################################################
                start = perturb_num*args.perturb_rate
                end = start + args.perturb_rate
                under_perturb_limit = torch.from_numpy((start < perturb_limit) | (end < perturb_limit))
                
                ################################################################################
                #########################  실질적인 perturb가 이뤄지는 부분  #########################
                ################################################################################
                if row.shape[0] == 1:
                    if torch.sum(under_perturb_limit.type(torch.IntTensor)) > 0:
                        images[row, R_argsort[:,range(start, end)]] = perturb_matrix 
                else:
                    if torch.sum(under_perturb_limit.type(torch.IntTensor)) > 0:
                        row_tmp = row[under_perturb_limit]
                        R_argsort_tmp = R_argsort[under_perturb_limit]
                        perturb_matrix = perturb_matrix[under_perturb_limit]
                        
                        images[row_tmp, R_argsort_tmp[:,range(start, end)]] = perturb_matrix 
                ################################################################################
                
                # perturb된 mask 외의 image부분을 앞서 저장했던 perturb안된 mask 외의 image부분으로 덮어 쓴다.
                if args.perturb_reference == 'mask' or args.perturb_reference == 'random':
                    images = images * new_mask + images_with_zero_on_mask
                
                if 'mnist' not in args.data_path:
                    widthNheight = int((images.shape[1]/depth)**(1/2))
                    images = images.reshape(images.shape[0], depth, widthNheight, widthNheight)
                
                ###################################################################################################
                ###################################  perturb후 변화하는 score 저장  ###################################
                ###################################################################################################
                args.interpreter = 'None'
                output, _, _ = model.forward(images, label, args, mask=new_mask) #### mask_modi was originally mask
                args.interpreter = 'lrp'
                
                if batch_num < 50:
                    general.save_hdf5(perturbed_result_path, 'image_{:02d}'.format(perturb_num), images[[0]].cpu().numpy())
                    general.save_hdf5(perturbed_result_path, 'output_{:02d}'.format(perturb_num), output.cpu().numpy())
                ###################################################################################################

            # measure elapsed time
            end = time.time()
        
        
        
def get_ROC_AUC(val_loader, model, criterion, mask, args):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        #################
        total_softmax = []
        total_label = []
        #################
        for i, bundle in enumerate(tqdm(val_loader)):
            ########################################
            if len(bundle) == 2:
                images, label = bundle
            elif len(bundle) == 3:
                images, label, mask = bundle
            
            images = images.type(torch.FloatTensor) 
            ########################################
            if args.arch[:5] == 'conv3':
                images = images.reshape(images.shape[0], 1, *images.shape[1:])
            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            
            #################################################################################################
            if args.mask_system == 'one_mask_per_class':
                mask_modi = torch.from_numpy(mask[label.cpu().numpy()]).type(torch.cuda.FloatTensor)
            elif args.mask_system in ['mask_with_seg', 'only_one_mask']:
                mask_modi = mask.type(torch.cuda.FloatTensor)
            else:
                mask_modi = None
            ##################################################################################################
            
            ############################################################################            
            args.interpreter = 'None'
            args.loss_type = 'None'
            args.lrp_target_layer = None
            
            output, lrp_loss, R = model.forward(images, label, args, mask_modi) #### mask_modi was originally mask
            label = label.cpu().numpy()
            
            softmax = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
            total_softmax.append(softmax)
            total_label.append(label)
            ############################################################################   
            
        total_softmax = np.concatenate(total_softmax, axis=0)
        total_label = np.concatenate(total_label, axis=0)
        
        ROC_path = os.path.join(args.result_path, 'ROC_AUC.hdf5')
        
        # get ROC
        fpr_tpr_thr = np.concatenate([fpr[np.newaxis,:], tpr[np.newaxis,:], threshold[np.newaxis,:]], axis=0)
        general.save_hdf5(ROC_path, 'fpr_tpr_thr', fpr_tpr_thr)
        
        # get AUC
        AUC = metrics.roc_auc_score(total_label, total_softmax[:,1])
        general.save_hdf5(ROC_path, 'auc', AUC)
        
        
        
class Dataset_train(Dataset):
    def __init__(self, path, args):
        self.f = h5py.File(path, 'r')
        self.keys = list(self.f.keys())
        if args.mask_data_size == 'None':
            self.mask_data_size = ''
        else:
            self.mask_data_size = args.mask_data_size
    
    def __len__(self):
        return len(self.f['train_data']) 
    
    def __getitem__(self, idx):
        image = self.f['train_data'][idx]
        label = self.f['train_data_label'][idx]
        mask_key = 'train_mask{}'.format(self.mask_data_size)
        if mask_key in self.keys:
            return image, label, self.f[mask_key][idx]
        return image, label

class Dataset_val(Dataset):
    def __init__(self, path, args):
        self.f = h5py.File(path, 'r')
        self.keys = list(self.f.keys())
        if args.mask_data_size == 'None':
            self.mask_data_size = ''
        else:
            self.mask_data_size = args.mask_data_size
    
    def __len__(self):
        return len(self.f['val_data']) 
    
    def __getitem__(self, idx):
        image = self.f['val_data'][idx]
        label = self.f['val_data_label'][idx]
        mask_key = 'val_mask{}'.format(self.mask_data_size)
        if mask_key in self.keys:
            return image, label, self.f[mask_key][idx]
        return image, label
    
class Dataset_test(Dataset):
    def __init__(self, path, args):
        self.f = h5py.File(path, 'r')
        self.keys = list(self.f.keys())
        if args.mask_data_size == 'None':
            self.mask_data_size = ''
        else:
            self.mask_data_size = args.mask_data_size
    
    def __len__(self):
        return len(self.f['test_data']) 
    
    def __getitem__(self, idx):
        image = self.f['test_data'][idx]
        label = self.f['test_data_label'][idx]
        mask_key = 'test_mask{}'.format(self.mask_data_size)
        if mask_key in self.keys:
            return image, label, self.f[mask_key][idx]
        return image, label
    
    
def get_perturb_limit(mask, shrink_size):
    mask = mask.cpu().numpy()
    original_size = mask.shape[-1]

    # mask가 나타내는 tumor의 영역보다 같은 모양의 조금 더 넓은 영역을 가진 mask로 수정
    mask_resized = general.resize(mask, (shrink_size, shrink_size))
    mask = general.resize(mask_resized, (original_size, original_size))
    mask = mask > 0

    mask = mask.reshape(mask.shape[0], -1)
    perturb_limit = np.sum(mask, axis=1)
    return perturb_limit


# def skip_interpretation(func):
#     def decorated(*args, **kwargs):
#         # Interpretation을 생략하기 위해 args를 강제로 다음과 같이 설정
#         ori_lambda_for_final, ori_interpreter = args.lambda_for_final, args.interpreter
#         args.lambda_for_final, args.interpreter = 0, 'None'
        
#         func(*args, **kwargs)
        
#         # args들 다시 원상복구
#         args.lambda_for_final, args.interpreter = ori_lambda_for_final, ori_interpreter
#         return func
#     return decorated


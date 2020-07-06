import os
import sys
sys.path.append('../utils')
import general
import help_main
import time
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy



def train(train_loader, model, epoch, criterion, optimizer, args):

    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
    label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        ###################################################
        if args.frame_mask:
#             print('mask :', mask.shape)
            mask = torch.zeros_like(mask).type(torch.FloatTensor).cuda(args.gpu)
            frame_width = 15/100
            width_small = int(mask.shape[2] * frame_width)
            width_big = mask.shape[2] - width_small
#             print(width_small, width_big)

            mask[:,:,:width_small,:] = 1.
            mask[:,:,width_big:,:] = 1.
            mask[:,:,:,:width_small] = 1.
            mask[:,:,:,width_big:] = 1.
        ###################################################
        
        ########################################################################
#         result_filename = os.path.join(args.result_path, 'test.hdf5')
#         general.save_hdf5(result_filename, 'img', img.cpu().detach().numpy())
#         general.save_hdf5(result_filename, 'label', label.cpu().detach().numpy())
        ########################################################################
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args) #### mask_modi was originally mask

        class_loss = criterion(output, label)
        train_loss = None
        if batch_idx % args.num_skip_cls_loss == 0:
            train_loss = class_loss + args.lambda_for_final * lrp_loss
        else:
            train_loss = args.lambda_for_final * lrp_loss

        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
#         label_list.append(label.cpu().data.numpy())
#         pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        
        
def train_gain(train_loader, model, epoch, criterion, optimizer, args):

    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
#     label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        ###################################################
        if args.frame_mask:
#             print('mask :', mask.shape)
            mask = torch.zeros_like(mask).type(torch.FloatTensor).cuda(args.gpu)
            frame_width = 15/100
            width_small = int(mask.shape[2] * frame_width)
            width_big = mask.shape[2] - width_small
#             print(width_small, width_big)

            mask[:,:,:width_small,:] = 1.
            mask[:,:,width_big:,:] = 1.
            mask[:,:,:,:width_small] = 1.
            mask[:,:,:,width_big:] = 1.
        ###################################################
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args) #### mask_modi was originally mask
        
        class_loss = criterion(output, label)
#         train_loss = class_loss + args.lambda_for_final * lrp_loss # + args.delta * gain_loss
        if batch_idx % args.num_skip_cls_loss == 0:
            train_loss = class_loss + args.lambda_for_final * lrp_loss # + args.delta * gain_loss
        else:
            train_loss = args.lambda_for_final * lrp_loss # + args.delta * gain_loss
        
        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
#         label_list.append(label.cpu().data.numpy())
#         pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
    
        optimizer.zero_grad()
        train_loss.backward(retain_graph=True)
        
        ####################################################################################
        # 여기서부터 gain 구조 적용
        masked_img_4_new_graph, masked_img, sig_mask = help_main.get_masked_img(img, R)
        interpreter = args.interpreter
        args.interpreter = 'None'
        new_output, _, _ = model.forward(masked_img_4_new_graph, label, mask, use_mask, args)
        args.interpreter = interpreter
        
        selected_scores = torch.gather(new_output, 1, label.unsqueeze(1))
        
        gain_loss = selected_scores.mean()
        masked_img_4_new_graph_grad = torch.autograd.grad(gain_loss*args.delta, masked_img_4_new_graph)[0]
#         if batch_idx % 5 == 0:
        masked_img.backward(masked_img_4_new_graph_grad)
        optimizer.step()
        ####################################################################################
        
        
        
        
        
def train_gain_no_grad_skip(train_loader, model, epoch, criterion, optimizer, args):

    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
#     label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        ###################################################
        if args.frame_mask:
#             print('mask :', mask.shape)
            mask = torch.zeros_like(mask).type(torch.FloatTensor).cuda(args.gpu)
            frame_width = 15/100
            width_small = int(mask.shape[2] * frame_width)
            width_big = mask.shape[2] - width_small
#             print(width_small, width_big)

            mask[:,:,:width_small,:] = 1.
            mask[:,:,width_big:,:] = 1.
            mask[:,:,:,:width_small] = 1.
            mask[:,:,:,width_big:] = 1.
        ###################################################
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args) #### mask_modi was originally mask
        
        class_loss = criterion(output, label)
#         train_loss = class_loss + args.lambda_for_final * lrp_loss # + args.delta * gain_loss
        if batch_idx % args.num_skip_cls_loss == 0:
            train_loss = class_loss + args.lambda_for_final * lrp_loss # + args.delta * gain_loss
        else:
            train_loss = args.lambda_for_final * lrp_loss # + args.delta * gain_loss
        
        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
#         label_list.append(label.cpu().data.numpy())
#         pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
    
        optimizer.zero_grad()
        train_loss.backward(retain_graph=True)
        
        ####################################################################################
        # 여기서부터 gain 구조 적용
        masked_img, sig_mask = help_main.get_masked_img_no_grad_skip(img, R)
        interpreter = args.interpreter
        args.interpreter = 'None'
        new_output, _, _ = model.forward(masked_img, label, mask, use_mask, args)
        args.interpreter = interpreter
        
        selected_scores = torch.gather(new_output, 1, label.unsqueeze(1))
        
        gain_loss = selected_scores.mean()
#         if batch_idx % 5 == 0:
        weighted_gain_loss = args.delta * gain_loss
        weighted_gain_loss.backward()
        optimizer.step()
        ####################################################################################
        
        
        
        
        
def train_grad_align_gain(train_loader, model, epoch, criterion, optimizer, args):
    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
    label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        # update전 param 저장
        weights_before = deepcopy(model.state_dict())
        
        ################### 첫번째 update ###################
        # interpretation을 skip하기 위해 args 강제 설정
        ori_lambda_for_final, ori_interpreter = args.lambda_for_final, args.interpreter
        args.lambda_for_final, args.interpreter = 0, 'None'
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)

        # args들 원상 복구
        args.lambda_for_final, args.interpreter = ori_lambda_for_final, ori_interpreter
        
        # 우선 class loss로만 update함
        class_loss = criterion(output, label)
        
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        ###################################################
        
        ################### 두번째 update ###################
        # lrp_loss를 구하기 위해 다시 forward
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)
        
        # lrp_loss로만 update함
        lrp_loss_weighted = args.lambda_for_final * lrp_loss
        
        optimizer.zero_grad()
        lrp_loss_weighted.backward(retain_graph=True)
        
        
        
        
        
        ####################################################################################
        # 여기서부터 gain 구조 적용
        masked_img_4_new_graph, masked_img, sig_mask = help_main.get_masked_img(img, R)
        interpreter = args.interpreter
        args.interpreter = 'None'
        new_output, _, _ = model.forward(masked_img_4_new_graph, label, mask, use_mask, args)
        args.interpreter = interpreter
        
        selected_scores = torch.gather(new_output, 1, label.unsqueeze(1))
        
        gain_loss = selected_scores.mean()
        masked_img_4_new_graph_grad = torch.autograd.grad(gain_loss*args.delta, masked_img_4_new_graph)[0]
        masked_img.backward(masked_img_4_new_graph_grad)
        ####################################################################################
        
        
        
        
        
        
        optimizer.step()
        
        # 두 update 끝난 후 param 저장
        weights_after = model.state_dict()
        ###################################################
        
        
        model.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * args.gamma) for name in weights_before})
        
        train_loss = class_loss.detach() + args.lambda_for_final * lrp_loss_weighted.detach()
        
        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
        label_list.append(label.cpu().data.numpy())
        pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
            
            
            
            
def train_grad_align_gain_inv(train_loader, model, epoch, criterion, optimizer, args):
    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
    label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        # update전 param 저장
        weights_before = deepcopy(model.state_dict())
        
        ################### 첫번째 update ###################
        # lrp_loss를 구하기 위해 다시 forward
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)
        
        # lrp_loss로만 update함
        lrp_loss_weighted = args.lambda_for_final * lrp_loss
        
        optimizer.zero_grad()
        lrp_loss_weighted.backward(retain_graph=True)
        
        ####################################################################################
        # 여기서부터 gain 구조 적용
        masked_img_4_new_graph, masked_img, sig_mask = help_main.get_masked_img(img, R)
        interpreter = args.interpreter
        args.interpreter = 'None'
        new_output, _, _ = model.forward(masked_img_4_new_graph, label, mask, use_mask, args)
        args.interpreter = interpreter
        
        selected_scores = torch.gather(new_output, 1, label.unsqueeze(1))
        
        gain_loss = selected_scores.mean()
        masked_img_4_new_graph_grad = torch.autograd.grad(gain_loss*args.delta, masked_img_4_new_graph)[0]
        masked_img.backward(masked_img_4_new_graph_grad)
        ####################################################################################
        
        optimizer.step()
        ###################################################
        
        
        ################### 두번째 update ###################
        # interpretation을 skip하기 위해 args 강제 설정
        ori_lambda_for_final, ori_interpreter = args.lambda_for_final, args.interpreter
        args.lambda_for_final, args.interpreter = 0, 'None'
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)

        # args들 원상 복구
        args.lambda_for_final, args.interpreter = ori_lambda_for_final, ori_interpreter
        
        # class loss로만 update함
        class_loss = criterion(output, label)
        
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        
        # 두 update 끝난 후 param 저장
        weights_after = model.state_dict()
        ###################################################
        
        
        
        
        model.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * args.gamma) for name in weights_before})
        
        train_loss = class_loss.detach() + args.lambda_for_final * lrp_loss_weighted.detach()
        
        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
        label_list.append(label.cpu().data.numpy())
        pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
     
        
        
        
def train_grad_align(train_loader, model, epoch, criterion, optimizer, args):

    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
    label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        # update전 param 저장
        weights_before = deepcopy(model.state_dict())
        
        ################### 첫번째 update ###################
        # interpretation을 skip하기 위해 args 강제 설정
        ori_lambda_for_final, ori_interpreter = args.lambda_for_final, args.interpreter
        args.lambda_for_final, args.interpreter = 0, 'None'
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)

        # args들 원상 복구
        args.lambda_for_final, args.interpreter = ori_lambda_for_final, ori_interpreter
        
        # 우선 class loss로만 update함
        class_loss = criterion(output, label)
        
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        ###################################################
        
        ################### 두번째 update ###################
        # lrp_loss를 구하기 위해 다시 forward
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)
        
        # lrp_loss로만 update함
        lrp_loss_weighted = args.lambda_for_final * lrp_loss
        
        optimizer.zero_grad()
        lrp_loss_weighted.backward()
        optimizer.step()
        
        # 두 update 끝난 후 param 저장
        weights_after = model.state_dict()
        ###################################################
        
        
        model.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * args.gamma) for name in weights_before})
        
        train_loss = class_loss.detach() + args.lambda_for_final * lrp_loss_weighted.detach()
        
        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
        label_list.append(label.cpu().data.numpy())
        pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
            
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
        
        
        
def train_grad_align2(train_loader, model, epoch, criterion, optimizer, args):

    # switch to train mode
    model.train()
    
    list_trn_loss, list_cls_loss, list_lrp_loss = [], [], []
    label_list, pred_list = [], []
    
    t = tqdm(train_loader)
    for batch_idx, (data, label, use_mask) in enumerate(t):
        
        img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
        label, use_mask = label.cuda(args.gpu), use_mask.cuda(args.gpu)
        img.requires_grad = True
        
        # update전 param 저장
        weights_before = deepcopy(model.state_dict())
        
        ################### 첫번째 update ###################
        # interpretation을 skip하기 위해 args 강제 설정
        ori_lambda_for_final, ori_interpreter = args.lambda_for_final, args.interpreter
        args.lambda_for_final, args.interpreter = 0, 'None'
        
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)

        # args들 원상 복구
        args.lambda_for_final, args.interpreter = ori_lambda_for_final, ori_interpreter
        
        # 우선 class loss로만 update함
        class_loss = criterion(output, label)
        
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        ###################################################
        
        ################### 두번째 update ###################
        # 첫번째 lrp_loss update
        # lrp_loss를 구하기 위해 다시 forward
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)
        
        # lrp_loss로만 update함
        lrp_loss_weighted = args.lambda_for_final * lrp_loss
        
        optimizer.zero_grad()
        lrp_loss_weighted.backward()
        optimizer.step()
        
        # 두번째 lrp_loss update
        # lrp_loss를 구하기 위해 다시 forward
        output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)
        
        # lrp_loss로만 update함
        lrp_loss_weighted = args.lambda_for_final * lrp_loss
        
        optimizer.zero_grad()
        lrp_loss_weighted.backward()
        optimizer.step()
        
        # 두 update 끝난 후 param 저장
        weights_after = model.state_dict()
        ###################################################
        
        
        model.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * args.gamma) for name in weights_before})
        
        train_loss = class_loss.detach() + args.lambda_for_final * lrp_loss_weighted.detach()
        
        list_trn_loss.append(train_loss.detach().cpu().numpy())
        list_cls_loss.append(class_loss.detach().cpu().numpy())
        list_lrp_loss.append(lrp_loss.detach().cpu().numpy())
        label_list.append(label.cpu().data.numpy())
        pred_list.append(output.cpu().data.numpy())
        
        t.set_description((
        ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
            np.mean(list_cls_loss),
            np.mean(list_lrp_loss),
            np.mean(list_trn_loss))))
        
        
        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            import IPython; IPython.embed()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

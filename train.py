# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, Evaluate_Generator
from time import time
from common.utils import deterministic_random

args = parse_args()
print(args)

seed = 4321
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
gpu_list = [0]

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

filter_widths = [int(x) for x in args.architecture.split(',')]
num_joints_in = 17
in_features = 2
num_joints_out = 17

if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos = RIEModel(num_joints_in, in_features,
                               num_joints_out,
                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                               channels=args.channels, latten_features=args.latent_features_dim,
                               dense=args.dense, is_train=True, Optimize1f=True, stage=args.stage)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos = RIEModel(num_joints_in, in_features,
                               num_joints_out,
                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                               channels=args.channels,
                               latten_features=args.latent_features_dim, dense=args.dense, is_train=True,
                               stage=args.stage)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

print('Loading dataset...')
dataset3d_path = 'data/data_3d_' + args.dataset + '.npz'
dataset2d_path = 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset_train = Human36mDataset(dataset3d_path,dataset2d_path,chunk_length=args.stride,augment=args.data_augmentation,pad=pad,causal_shift=causal_shift,is_train=True)
    dataset_test = Human36mDataset(dataset3d_path,dataset2d_path,chunk_length=args.stride,augment=False,pad=pad,causal_shift=causal_shift,is_train=False)
else:
    raise KeyError('Invalid dataset')

if torch.cuda.is_available():
    # model_pos = model_pos.cuda()
    model_pos = torch.nn.DataParallel(model_pos, device_ids=gpu_list).cuda()

if args.pretrain:
    pretrain_filename = os.path.join(args.checkpoint, args.pretrain)
    print('Loading pretrain model', pretrain_filename)
    checkpoint_p = torch.load(pretrain_filename, map_location=lambda storage, loc: storage)
    pretrain_dict = checkpoint_p['model_pos']

    model_dict = model_pos.state_dict()

    state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict.keys()}

    state_dict = {k: v for i, (k, v) in enumerate(state_dict.items()) if i < 317}

    model_dict.update(state_dict)
    model_pos.load_state_dict(model_dict)

    cnt = 0
    for name, value in model_pos.named_parameters():

        if cnt < 167:
            value.requires_grad = False
        cnt = cnt + 1

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'])

print('INFO: Testing on {} frames'.format(len(dataset_test)))

if not args.evaluate:
    ## train
    trainDataLoader = DataLoader(dataset_train, batch_size=args.batch_size * len(gpu_list),shuffle=True,num_workers=6,pin_memory=True)
    testDataLoader = DataLoader(dataset_test, batch_size=args.batch_size * len(gpu_list),shuffle=False,pin_memory=True)
    lr = args.learning_rate
    optimizer = optim.Adam(model_pos.parameters(), lr=lr, amsgrad=True)
    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        batch_ptr = 0
        N = 0
        N_semi = 0
        model_pos.train()

        # Regular supervised scenario
        for label,_, inputs_3d, inputs_2d in tqdm(trainDataLoader):
            inputs_3d = inputs_3d.cuda(non_blocking=True)

            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            loss_3d_pos = mpjpe_loss(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()
            if N > 10000:
                break

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0
            eval_time = time()
            batch_ptr = 0

            if not args.no_eval:
                # Evaluate on test set
                for label,_, inputs_3d, inputs_2d in tqdm(testDataLoader):
                    inputs_3d = inputs_3d.cuda(non_blocking=True)
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe_loss(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for label,_, inputs_3d, inputs_2d in tqdm(trainDataLoader):
                    if inputs_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue
                    # if torch.cuda.is_available():
                    #     inputs_3d = inputs_3d.cuda()
                    #     inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe_loss(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

        elapsed = (time() - start_time) / 60
        eval_elapsed = (time() - eval_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:

            print('[%d] time %.2f eval_time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                epoch + 1,
                elapsed,
                eval_elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            f = open(log_path, mode='a')
            f.write('[%d] time %.2f eval_time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f\n' % (
                epoch + 1,
                elapsed,
                eval_elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))
            f.close()

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        model_pos.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'stage_' + str(args.stage) + '_epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
            }, chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')

# Evaluate
def evaluate(test_dataloader, return_predictions=False):
    epoch_loss_3d_pos = {}

    with torch.no_grad():
        model_pos.eval()
        N = 0
        cnt = 0
        dataSet = test_dataloader.dataset
        joints_left = dataSet.joints_left
        joints_right = dataSet.joints_right
        kps_left = dataSet.kps_left
        kps_right = dataSet.kps_right

        batch_num = len(test_dataloader)
        batch_size = test_dataloader.batch_size

        output = torch.zeros([batch_num,batch_size, 17, 3], dtype=torch.float32)

        if args.test_time_augmentation:
            for label,_, inputs_3d, inputs_2d in tqdm(test_dataloader):

                # Flip 2D keypoints
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[:,:, :, 0] *= -1
                inputs_2d_flip[:,:, kps_left + kps_right] = inputs_2d_flip[:,:, kps_right + kps_left]

                inputs_3d = inputs_3d.cuda(non_blocking=True)

                # Positional model
                predicted_3d_pos = model_pos(inputs_2d)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,joints_right + joints_left]

                predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,keepdim=True)

                if return_predictions:
                    output[cnt] = predicted_3d_pos.squeeze().cpu()
                    cnt = cnt + 1
                    continue

                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                inputs_3d[:, :, 0] = 0
                error = mpjpe(predicted_3d_pos, inputs_3d)

                # epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
                # N += inputs_3d.shape[0] * inputs_3d.shape[1]
                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                error_p = p_mpjpe(predicted_3d_pos,inputs)
                # epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos,inputs)
                for index,subject_action_camIndex in enumerate(label):
                    key = subject_action_camIndex.split("_")[1].split(" ")[0] if not args.by_subject else subject_action_camIndex.split("_")[0]
                    if key in epoch_loss_3d_pos:
                        epoch_loss_3d_pos[key]["mpjpe"] += error[index]
                        epoch_loss_3d_pos[key]["p-mpjpe"] += error_p[index]
                        epoch_loss_3d_pos[key]["count"] += 1
                    else:
                        epoch_loss_3d_pos[key] = {"mpjpe":error[index],"p-mpjpe":error_p[index],"count":1}
        else:
            for label,_, inputs_3d, inputs_2d in test_dataloader:
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()

                # Positional model
                predicted_3d_pos = model_pos(inputs_2d)

                if return_predictions:
                    return predicted_3d_pos.squeeze().cpu().numpy()
                    
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                inputs_3d[:, :, 0] = 0

                error = mpjpe(predicted_3d_pos, inputs_3d)

                # epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]
                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                error_p = p_mpjpe(predicted_3d_pos,inputs)
                # epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos,inputs)
                for index,subject_action_camIndex in enumerate(label):
                    key = subject_action_camIndex.split("_")[1].split(" ")[0] if not args.by_subject else subject_action_camIndex.split("_")[0]
                    if key in epoch_loss_3d_pos:
                        epoch_loss_3d_pos[key]["mpjpe"] += error[index]
                        epoch_loss_3d_pos[key]["p-mpjpe"] += error_p[index]
                        epoch_loss_3d_pos[key]["count"] += 1
                    else:
                        epoch_loss_3d_pos[key] = {"mpjpe":error[index],"p-mpjpe":error_p[index],"count":1}

        if return_predictions:
            return output.numpy()

    log_path = os.path.join(args.checkpoint, 'evaluating_log.txt')
    f = open(log_path, mode='a')
    e1_list = []
    e2_list = []
    for key, errorInfo in epoch_loss_3d_pos.items():
        if key is None:
            print('----------')
            f.write('----------\n')
        else:
            print('----' + key + '----\n')
            f.write('----' + key + '----\n')
        e1 = (errorInfo["mpjpe"] / errorInfo["count"]) * 1000
        e2 = (errorInfo["p-mpjpe"] /errorInfo["count"]) * 1000
        print('Test time augmentation:', args.test_time_augmentation)
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('----------')

        f.write('Test time augmentation:' + str(args.test_time_augmentation) + '\n')
        f.write('Protocol #1 Error (MPJPE):' + str(e1) + 'mm\n')
        f.write('Protocol #2 Error (P-MPJPE):' + str(e2) + 'mm\n')
        f.write('----------\n')
        e1_list.append(e1)
        e2_list.append(e2)
    
    f.close()
    return e1_list, e2_list

if args.render:
    pass
else:
    def run_evaluation():

        testDataLoader = DataLoader(dataset_test, batch_size=args.batch_size*len(gpu_list),shuffle=False,pin_memory=True)
        errors_p1, errors_p2 = evaluate(testDataLoader)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')

        log_path = os.path.join(args.checkpoint, 'evaluating_log.txt')
        f = open(log_path, mode='a')
        f.write('Protocol #1   (MPJPE) action-wise average:' + str(round(np.mean(errors_p1), 1)) + 'mm\n')
        f.write('Protocol #2 (P-MPJPE) action-wise average:' + str(round(np.mean(errors_p2), 1)) + 'mm\n')
        f.close()

    if not args.by_subject:
        run_evaluation()
    else:
        run_evaluation()

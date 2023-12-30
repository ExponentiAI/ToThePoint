from __future__ import print_function

import wandb
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from util import IOStream, AverageMeter,cal_loss
import os
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util import NT_Xent,accuracy
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapNetDataLoaderContrastive import ModelNet40SVM,ScanObjectNNSVM,ShapeNetConrastiveLess
from models.dgcnn import DGCNN
import random
os.environ["WANDB_MODE"] = "offline"

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu',  default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='dgcnn_cls', help='model name [default: dgcnn_cls]')
    parser.add_argument('--num_category', default=40, type=int, help='training num_category')
    parser.add_argument('--epoch', default=800, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--pretrain', type=bool, default=True, help='Point Number')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--exp_name', type=str, default='dgcnn_shapenet_contrastive_less_SVM', help='use uniform sampiling ')
    parser.add_argument('--temperature', type=float, default= 0.1, help='temperature of infoNCE loss')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--seed', type=float, default= 42, help='the seed of random')
    parser.add_argument('--interval', type=int, default= 50, help='the interval to save the model')
    return parser.parse_args()

def test(model, train_loader,test_loader):

    classifier = model.eval()

    feats_train = []
    labels_train = []

    for i, (points_1, label) in enumerate(train_loader):

        labels = label.numpy().tolist()
        if not args.use_cpu:
            points = points_1.cuda()

        points = points.transpose(2, 1)
        feats = classifier(points)[-1]
        feats = feats.detach().cpu().numpy()

        for feat in feats:
            feats_train.append(feat)
        labels_train += labels

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)

    feats_test = []
    labels_test = []

    for j, (points_1, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        labels = target.numpy().tolist()
        if not args.use_cpu:
            points= points_1.cuda()

        points = points.transpose(2, 1)
        feats = classifier(points)[-1]
        feats = feats.detach().cpu().numpy()

        for feat in feats:
            feats_test.append(feat)
        labels_test += labels

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)

    model_tl = SVC(C=0.1, kernel='linear')
    model_tl.fit(feats_train, labels_train)
    test_accuracy = model_tl.score(feats_test, labels_test)
    print(f"Linear Accuracy : {test_accuracy}")

    return test_accuracy


def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    train_dataset = ShapeNetConrastiveLess()
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_val_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=1024), batch_size=128, shuffle=True, num_workers=2)
    test_val_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=1024), batch_size=128, shuffle=True, num_workers=2)
    


    '''MODEL LOADING'''
    wandb.init(project="tothepooint", name=args.exp_name)
    num_class = args.num_category
    model = DGCNN(args = args,output_channels = num_class)

    wandb.watch(model)

    criterion = NT_Xent(args.batch_size,args.temperature,world_size=1)
    model.apply(inplace_relu)

    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()


    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate*100, momentum=0.9)

    global_epoch = 0
    global_step = 0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    train_loss = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        wandb_log = {}
        model = model.train()

        for batch_id, (points_1,points_2) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            batch_size = points_1.size()[0]
            points_1 = points_1.transpose(2, 1)
            points_2 = points_2.transpose(2, 1)

            if not args.use_cpu:
                points_1,points_2 = points_1.cuda(), points_2.cuda()

            x11_prototype, x12_prototype, _= model(points_1)
            x21_prototype, x22_prototype, _= model(points_2)

            contrastive_loss_1,logits_1, labels_1 = criterion(x11_prototype, x12_prototype)
            contrastive_loss_12, logits_12, labels_12 = criterion(x11_prototype, x21_prototype)
            contrastive_loss_2, logits_2, labels_2 = criterion(x21_prototype, x22_prototype)

            logits = torch.cat((logits_1,logits_12,logits_2),dim=0)
            labels = torch.cat((labels_1, labels_12, labels_2), dim=0)
            loss = contrastive_loss_1 + contrastive_loss_12 + contrastive_loss_2

            accuracy1, accuracy5 = accuracy(logits, labels, topk=(1, 5))
            acc1.update(accuracy1.item(), batch_size)
            acc5.update(accuracy5.item(), batch_size)

            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss.update(loss.item(), batch_size)


        wandb_log['Train Loss'] = train_loss.avg
        wandb_log['acc1'] = acc1.avg
        wandb_log['acc5'] = acc5.avg
        save_best_file = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model_.pth'.format(epoch=epoch))
        save_interval_file = os.path.join(f'checkpoints/{args.exp_name}/models/', '{}_model.pth'.format(epoch+1))

        with torch.no_grad():
            class_acc = test(model.eval(),train_val_loader ,test_val_loader)

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                best_epoch = epoch + 1
            log_string('Test Liear Accuracy: %f' % ( class_acc))
            log_string('Best Liear Accuracy: %f' % (best_class_acc))

            if (class_acc >= best_class_acc):
                logger.info('Save model...')
                log_string('Saving at %s' % save_best_file)
                torch.save(model.state_dict(), save_best_file)
            if (epoch+1) % args.interval == 0:
                torch.save(model.state_dict(), save_interval_file)
            global_epoch += 1
            wandb_log['Liear Accuracy'] = class_acc
            wandb_log['Epoch'] = epoch
        wandb.log(wandb_log)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
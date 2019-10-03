import os
import sys
from os.path import join as join
import pickle
import pprint
import random
import tqdm
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import lib.utils as utils
from lib.config import cfg
from collections import OrderedDict
import datasets.data_loader as data_loader

import models
import torch.multiprocessing as mp
import torch.distributed as dist
from os.path import join as join


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_loader(self):
        self.train_loader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
            join(cfg.DATA_LOADER.DATA_ROOT, cfg.DATA_LOADER.LIST, cfg.DATA_LOADER.FOLDER + '_train.txt'), 
            use_mirror=self.args.mirror)

        self.test_loader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT,
            join(cfg.DATA_LOADER.DATA_ROOT, cfg.DATA_LOADER.LIST, cfg.DATA_LOADER.FOLDER + '_test.txt'),
            use_mirror=self.args.mirror)

    def snapshot_path(self, name, epoch):
        path = cfg.ROOT_DIR
        pos = path.rfind('/')
        path = path[0:pos]
        snapshot_folder = join(path, 'snapshot')
        return join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def load_checkpoint(self, netG, netE):
        if self.args.resume > 0:
            netG_dict = torch.load(self.snapshot_path("netG", self.args.resume), map_location=lambda storage, loc: storage)
            current_state = netG.state_dict()
            keys = list(current_state.keys())
            for key in keys:

                current_state[key] = netG_dict['module.' + key]
            netG.load_state_dict(current_state)

            netE_dict = torch.load(self.snapshot_path("netE", self.args.resume), map_location=lambda storage, loc: storage)
            current_state = netE.state_dict()
            keys = list(current_state.keys())
            for key in keys:
                current_state[key] = netE_dict['module.' + key]
            netE.load_state_dict(current_state)

    def init_network(self):
        netG = models.__dict__[cfg.MODEL.NET](pretrained=True)
        netE = models.classifier.Classifier(class_num=cfg.MODEL.CLASS_NUM, distributed=self.distributed).cuda()

        self.load_checkpoint(netG, netE)
        if self.distributed:
            sync_netG = netG  #nn.SyncBatchNorm.convert_sync_batchnorm(netG)
            sync_netE = netE  #nn.SyncBatchNorm.convert_sync_batchnorm(netE)
            self.netG = torch.nn.parallel.DistributedDataParallel(sync_netG.to(self.device),
                device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.netE = torch.nn.parallel.DistributedDataParallel(sync_netE.to(self.device),
                device_ids=[self.args.local_rank], output_device=self.args.local_rank)
        else:
            self.netG = torch.nn.DataParallel(netG).cuda()
            self.netE = torch.nn.DataParallel(netE).cuda()

    def display_dim(self, train_feats, train_labels, test_feats, test_labels):
        if (self.distributed == True) and (dist.get_rank() != 0):
            return
        self.logger.info('shape info train_feats: {}, train_labels: {}'.format(train_feats.shape, train_labels.shape))
        self.logger.info('shape info test_feats: {}, test_labels: {}'.format(test_feats.shape, test_labels.shape))

    def eval(self, data_loader):
        self.netG.eval()
        self.netE.eval()

        correct = 0
        tick = 0
        subclasses_correct = np.zeros(cfg.MODEL.CLASS_NUM)
        subclasses_tick = np.zeros(cfg.MODEL.CLASS_NUM)

        # data set
        n_samples = data_loader.dataset.__len__()
        feats = torch.zeros(n_samples, cfg.MODEL.IN_DIM).cuda()
        labels = np.zeros((n_samples,), dtype='int')

        probs = torch.zeros(n_samples, cfg.MODEL.CLASS_NUM)
        preds = np.zeros((n_samples,), dtype='int')

        with torch.no_grad():
            index = 0
            for imgs, gtlabels in tqdm.tqdm(data_loader):
                batch_size = imgs.size(0)
                imgs = Variable(imgs.cuda())
                _, _unsup_pool5_out = self.netG(imgs)
                _, _unsup_logits_out = self.netE(_unsup_pool5_out)
                prob = F.softmax(_unsup_logits_out, dim=1)
                pred = prob.data.cpu().numpy().argmax(axis=1)

                feats[index:index + batch_size, :] = _unsup_pool5_out.data
                labels[index:index + batch_size] = gtlabels.data.cpu().numpy()
                probs[index:index + batch_size, :] = prob.data
                preds[index:index + batch_size] = pred

                index += batch_size
                gtlabels = gtlabels.numpy()
                for i in range(pred.size):
                    subclasses_tick[gtlabels[i]] += 1
                    if pred[i] == gtlabels[i]:
                        correct += 1
                        subclasses_correct[pred[i]] += 1
                    tick += 1

        feats = feats.data.cpu().numpy()
        probs = probs.data.cpu().numpy()

        #correct = correct * 1.0 / tick
        #subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        #mean_class_acc = subclasses_result.mean()
        #zeors_num = subclasses_result[subclasses_result == 0].shape[0]
        #mean_acc_str = "*** mean class acc = {:.2%}, overall = {:.2%}, missing = {:d}".format(mean_class_acc, correct, zeors_num)
        #print(mean_acc_str)

        return feats, labels, probs, preds

    def compute_feats(self):
        train_feats, train_labels, train_probs, train_preds = self.eval(self.train_loader)
        test_feats, test_labels, _, _ = self.eval(self.test_loader)

        self.display_dim(train_feats, train_labels, test_feats, test_labels)

        if (self.distributed == False) or (dist.get_rank() == 0):
            if self.args.mirror == False:
                result_folder = os.path.join(cfg.ROOT_DIR, 'result')
            else:
                result_folder = os.path.join(cfg.ROOT_DIR, 'result_mirror')
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            pickle.dump({'feats': train_feats, 'labels': train_labels}, open(join(result_folder, cfg.DATA_LOADER.FOLDER + '_train.pkl'), 'wb'))
            pickle.dump({'feats': test_feats, 'labels': test_labels}, open(join(result_folder, cfg.DATA_LOADER.FOLDER + '_test.pkl'), 'wb'))
            pickle.dump({'probs': train_probs, 'labels': train_preds }, open(join(result_folder, cfg.DATA_LOADER.FOLDER + '_probs.pkl'), 'wb'))

    def train(self):
        self.setup_logging()
        self.setup_loader()
        self.init_network()
        self.compute_feats()

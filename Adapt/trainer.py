import os
import sys
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
import losses
import evaluation
import torch.multiprocessing as mp
import torch.distributed as dist

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

        self.setup_logging()
        self.setup_loader()
        self.init_network()
        self.iteration = 0

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
        self.trg_train_listloader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT, \
            os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.TARGET + '_train.txt'))
        
        ################################### Source Target domain data loader  ###################################
        self.src_image_set = data_loader.load_src_trainset()
        self.trg_paths, _ = utils.loadlines(os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.TARGET + '_train.txt'))
        probs, plabels = utils.load_trg_plabels()
        trg_paths, self.trg_probs, self.trg_varlabels = utils.filter_trg_plabels(self.trg_paths, probs, plabels) 
         
        if cfg.DATA_LOADER.ITER == -1:
            if cfg.MODEL.SOURCE_ONLY == True:
                src_paths, _ = utils.loadlines(os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.SOURCE + '_train.txt'))
                cfg.DATA_LOADER.ITER = len(src_paths) // cfg.TRAIN.BATCH_SIZE
            else:
                cfg.DATA_LOADER.ITER = len(trg_paths) // cfg.TRAIN.BATCH_SIZE

        cls_info, self.src_train_loader = data_loader.load_mergesrc_train(self.distributed, self.src_image_set)
        self.trg_train_loader = data_loader.load_trg_train(self.distributed, trg_paths, self.trg_varlabels, cls_info)
        ########################################################################################################

        ########################################### load test loader ###########################################
        # Target test loader
        self.trg_test_loader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT, \
            os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.TARGET + '_test.txt'))

        self.trg_real_loader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT, \
            os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', 'real_test.txt'))
        ########################################################################################################

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)

        if epoch % cfg.SOLVER.SNAPSHOT_ITERS == 0:
            torch.save(self.netG.state_dict(), self.snapshot_path("netG", epoch))
            torch.save(self.netE.state_dict(), self.snapshot_path("netE", epoch))

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
        netE = models.classifier.Classifier(class_num=cfg.MODEL.CLASS_NUM, distributed=self.distributed)
        self.load_checkpoint(netG, netE)

        if self.distributed:
            #sync_netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG)
            #sync_netE = nn.SyncBatchNorm.convert_sync_batchnorm(netE)
            sync_netG = netG
            sync_netE = netE
            self.netG = torch.nn.parallel.DistributedDataParallel(sync_netG.to(self.device), 
                device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.netE = torch.nn.parallel.DistributedDataParallel(sync_netE.to(self.device),
                device_ids=[self.args.local_rank], output_device=self.args.local_rank)
        else:
            self.netG = torch.nn.DataParallel(netG).cuda()
            self.netE = torch.nn.DataParallel(netE).cuda()

        self.optim = models.optimizer.Optimizer(self.netG, self.netE)

        self.cross_ent = losses.create('CrossEntropy').cuda()
        if cfg.LOSSES.MMD_WEIGHT > 0:
            self.mmd = losses.create('MMD').cuda()
        if cfg.LOSSES.TPN_TASK_WEIGHT > 0:
            self.tpn_task = losses.create('TpnTask').cuda()
        if cfg.LOSSES.TRG_GXENT_WEIGHT > 0:
            self.trg_xent = losses.create('GeneralEntropy').cuda()
        if cfg.LOSSES.SYMM_XENT_WEIGHT > 0:
            self.symm_xent_loss = losses.create('SymmEntropy').cuda()

    def eval(self, epoch):
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            if (self.distributed == False) or (dist.get_rank() == 0):
                os.mkdir(result_folder)

        loaders = []
        names = []
        loaders.append(self.trg_test_loader)
        names.append(cfg.DATA_LOADER.TARGET)

        loaders.append(self.trg_real_loader)
        names.append('Real')

        for i, loader in enumerate(loaders):
            mean_acc, preds = evaluation.eval(epoch, names[i], loader, self.netG, self.netE)
            if mean_acc is not None:
                if (self.distributed == False) or (dist.get_rank() == 0):
                    #self.logger.info(mean_acc)
                    with open(os.path.join(result_folder, str(epoch) + '_' + names[i] + '.txt'), 'w') as fid:
                        for v in preds:
                            fid.write(str(v) + '\n')
            
    def train_mode(self):
        self.netG.train(mode=True)
        self.netE.train(mode=True)

    def display(self, iteration, losses, loss_arr, loss_w):
        if (self.distributed == True) and (dist.get_rank() != 0):
            return

        if iteration % cfg.SOLVER.DISPLAY == 0:
            self.logger.info('Iteration ' + str(iteration) + ', lr = ' +  str(self.optim.get_lr()) + ', loss = ' + str(losses.data.cpu().numpy()))
            for lidx, losses in enumerate(loss_arr):
                loss_name = losses[0]
                loss_value = losses[1]
                self.logger.info('  ' + loss_name + ' = ' + str(loss_value) \
                    + ' (* ' + str(loss_w[lidx]) + ' = ' + str(loss_value * loss_w[lidx]) + ')')

    def compute_trg_pseudolabels(self, epoch):
        if cfg.DATA_LOADER.COMPUTE_TRG_PSEUDOLABELS == False:
            return None, None
        
        probs, plabels = evaluation.evalprobs(self.trg_train_listloader, self.netG, self.netE)
        if self.distributed:
            probs = utils.broadcast_numpy(probs)
            plabels = utils.broadcast_numpy(plabels)
        
        ############################################## Output Target Pseudo labels ##############################################
        if cfg.DATA_LOADER.OUTPUT_TRG_PSEUDOLABELS == True:
            if (self.distributed == False) or (dist.get_rank() == 0):
                result_folder = os.path.join(cfg.ROOT_DIR, 'result')
                if not os.path.exists(result_folder):
                    os.mkdir(result_folder)
                pickle.dump({ 'probs': probs, 'labels': plabels }, \
                    open(os.path.join(result_folder, str(epoch) + '_' + cfg.DATA_LOADER.TARGET + '.pkl'), 'wb'))
        ##########################################################################################################################
        self.train_mode()
        return probs, plabels

    def train_domain_adapt(self, iteration, epoch, s_imgs, t_imgs, t_index, s_labels, t_labels): 
        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())
        t_imgs = Variable(t_imgs.cuda())
        t_index = Variable(t_index.cuda())
        t_labels = Variable(t_labels.cuda()) if self.trg_varlabels is not None else None

        self.optim.zero_grad()

        s = s_imgs.shape[0]
        data = Variable(torch.cat((s_imgs, t_imgs), 0))
        _, pool5_output = self.netG(data)
        features, logits_out = self.netE(pool5_output)

        sup_feats, sup_logits_out = features[:s, :], logits_out[:s, :]
        unsup_feats, unsup_logits_out = features[s:, :], logits_out[s:, :]

        #_, sup_pool5_out = self.netG(s_imgs)
        #sup_feats, sup_logits_out = self.netE(sup_pool5_out)

        #_, unsup_pool5_out = self.netG(t_imgs)
        #unsup_feats, unsup_logits_out = self.netE(unsup_pool5_out)

        #if self.distributed:
        #    s_labels = utils.sync_labels(s_labels)
        #    t_labels = utils.sync_labels(t_labels)
        #    sup_logits_out = utils.sync_tensor(sup_logits_out)
        #    unsup_logits_out = utils.sync_tensor(unsup_logits_out)
        #    sup_feats = utils.sync_tensor(sup_feats)
        #    unsup_feats = utils.sync_tensor(unsup_feats)

        loss_arr = []
        loss_w = []

        # source cross entropy loss
        loss, loss_info = self.cross_ent(sup_logits_out, s_labels)
        loss_arr.append(loss_info)
        loss_w.append(cfg.LOSSES.CROSS_ENT_WEIGHT)

        # mmd loss
        if cfg.LOSSES.MMD_WEIGHT > 0:
            mmd, loss_info = self.mmd(sup_feats, unsup_feats, s_labels, t_labels)
            loss_arr.append(loss_info)
            loss_w.append(cfg.LOSSES.MMD_WEIGHT)
            loss += mmd * cfg.LOSSES.MMD_WEIGHT

        # tpn task specific loss
        if cfg.LOSSES.TPN_TASK_WEIGHT > 0:
            tpn_task, loss_info = self.tpn_task(sup_feats, unsup_feats, s_labels, t_labels)
            loss_arr.append(loss_info)
            loss_w.append(cfg.LOSSES.TPN_TASK_WEIGHT)
            loss += tpn_task * cfg.LOSSES.TPN_TASK_WEIGHT

        # target general classification loss
        if cfg.LOSSES.TRG_GXENT_WEIGHT > 0:
            trg_xent, loss_info = self.trg_xent(unsup_logits_out, t_labels)
            loss_arr.append(loss_info)
            loss_w.append(cfg.LOSSES.TRG_GXENT_WEIGHT)
            loss += trg_xent * cfg.LOSSES.TRG_GXENT_WEIGHT

        if cfg.LOSSES.SYMM_XENT_WEIGHT > 0:
            symm_xent, loss_info = self.symm_xent_loss(unsup_logits_out, t_labels)
            loss_arr.append(loss_info)
            loss_w.append(cfg.LOSSES.SYMM_XENT_WEIGHT)
            loss += symm_xent * cfg.LOSSES.SYMM_XENT_WEIGHT

        self.display(iteration, loss, loss_arr, loss_w)

        #if self.distributed:
        #    loss *= dist.get_world_size()

        loss.backward()
        self.optim.step(epoch)
        self.iteration += 1

    def train(self):
        for epoch in range(0, cfg.SOLVER.MAX_EPOCH):
            self.eval(epoch)
            self.train_mode()
            
            source_loader_iter = iter(self.src_train_loader)
            target_loader_iter = iter(self.trg_train_loader)

            for t_index, t_imgs, t_labels in target_loader_iter:
                try:
                    _, s_imgs, s_labels = source_loader_iter.next()
                    self.train_domain_adapt(self.iteration, epoch, s_imgs, t_imgs, t_index, s_labels, t_labels)
                except StopIteration:
                    _, self.src_train_loader = data_loader.load_mergesrc_train(self.distributed, self.src_image_set)
                
            self.save_model(epoch + 1)
            cls_info, self.src_train_loader = data_loader.load_mergesrc_train(self.distributed, self.src_image_set)

            if epoch > cfg.MODEL.UPDATE_LABEL:
                self.logger.info('Update labels at epoch:' + str(epoch))
                ################### Refresh the target labels ###################
                probs, plabels = self.compute_trg_pseudolabels(epoch + 1)
                trg_paths, self.trg_probs, self.trg_varlabels = utils.filter_trg_plabels(self.trg_paths, probs, plabels)
                #################################################################
                self.trg_train_loader = data_loader.load_trg_train(self.distributed, trg_paths, self.trg_varlabels, cls_info)
            
    def train_src_only(self):
        for epoch in range(0, cfg.SOLVER.MAX_EPOCH):
            self.eval(epoch)
            self.train_mode()
            
            for index, imgs, labels in self.src_train_loader:
                imgs = Variable(imgs.cuda())
                labels = Variable(labels.cuda())
                self.optim.zero_grad()

                _, sup_pool5_out = self.netG(imgs)
                _, sup_logits_out = self.netE(sup_pool5_out)

                #if self.distributed:              
                #    labels = utils.sync_labels(labels)
                #    sup_logits_out = utils.sync_tensor(sup_logits_out)

                loss_arr = []
                loss_w = []

                # source cross entropy loss
                loss, loss_info = self.cross_ent(sup_logits_out, labels)
                loss_arr.append(loss_info)
                loss_w.append(cfg.LOSSES.CROSS_ENT_WEIGHT)

                self.display(self.iteration, loss, loss_arr, loss_w)

                #if self.distributed:
                #    loss *= dist.get_world_size()

                loss.backward()
                self.optim.step(epoch)
                self.iteration += 1

            self.save_model(epoch + 1)
            _, self.src_train_loader = data_loader.load_mergesrc_train(self.distributed, self.src_image_set)
            self.compute_trg_pseudolabels(epoch + 1)

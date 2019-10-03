import os
import sys
import argparse
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
from lib.config import cfg, cfg_from_file
from collections import OrderedDict
from datasets.pkl_trg_dataset import PklTrgDataset
import torch.multiprocessing as mp
import torch.distributed as dist
import lib.utils as utils
import models
import optimizer
import losses
import datasets.data_loader as data_loader


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
        self.load_data()
        self.iteration = 0

        self.models = []
        self.optims = []
        self.names = []
        for i in range(len(cfg.MODEL.NETS)):
            in_dim = utils.get_dim(i)
            model = models.create(cfg.MODEL.NETS[i], in_dim=in_dim, out_dim=cfg.MODEL.EMBED_DIM[i]).cuda()
            optim = optimizer.Optimizer(model)
            self.models.append(model)
            self.optims.append(optim)
            self.names.append(cfg.MODEL.NAMES[i])

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

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        #if not os.path.exists(snapshot_folder):
        #    os.mkdir(snapshot_folder)
        #if epoch % cfg.SOLVER.SNAPSHOT_ITERS == 0:
        #    torch.save(self.model.state_dict(), self.snapshot_path("model", epoch))

    def load_data(self):
        self.reltr_loader = data_loader.load_train('real', cfg.DATA_LOADER.SRC_CNT_PCLS)
        self.inftr_loader = data_loader.load_train('infograph', cfg.DATA_LOADER.SRC_CNT_PCLS)
        self.qdrtr_loader = data_loader.load_train('quickdraw', cfg.DATA_LOADER.SRC_CNT_PCLS)
        self.skttr_loader = data_loader.load_train('sketch', cfg.DATA_LOADER.SRC_CNT_PCLS)
        
        self.relte_loader = data_loader.load_test('real', 'test')
        self.trgte_loader = data_loader.load_test(cfg.DATA_LOADER.TARGET, 'test')
        self.trgte_tr_loader = data_loader.load_test(cfg.DATA_LOADER.TARGET, 'train')

        self.trgtr_dataset = PklTrgDataset(cfg.DATA_LOADER.TARGET, 'train', cfg.DATA_LOADER.TRG_CNT_PCLS)
        trgtr_probs, trgtr_labels = utils.load_probpkl(cfg.DATA_LOADER.TARGET)
        self.trgtr_dataset.filter_trg_plabels(trgtr_probs)
        self.trgtr_loader = data_loader.load_trg_train(self.trgtr_dataset)

        self.iters_epoch = int(self.reltr_loader.dataset.__len__() * 1.0 // (cfg.DATA_LOADER.SRC_CNT_PCLS * cfg.MODEL.CLASS_NUM)) * cfg.SOLVER.EPOCH_K

    def display(self, iteration, loss, loss_info):
        if (self.distributed == True) and (dist.get_rank() != 0):
            return
        if iteration % cfg.SOLVER.DISPLAY == 0:
            self.logger.info('Iteration ' + str(iteration) + ', lr = ' +  str(self.optims[0].get_lr()) + ', loss = ' + str(loss.data.cpu().numpy()))

            for key in sorted(loss_info):
                loss_name = key
                loss_value = loss_info[key]
                self.logger.info('  ' + loss_name + ' = ' + str(loss_value))

    def eval_one(self, test_loader):
        dim = cfg.MODEL.CLASS_NUM
        n_samples = test_loader.dataset.__len__()
        probs_arr = []
        glabels = np.zeros((n_samples,), dtype='int')
        for _ in range(len(self.models)):
            probs_arr.append(np.zeros((n_samples, dim)))

        with torch.no_grad():
            index = 0
            for feats, labels in test_loader:
                feats = [feat.cuda() for feat in feats]
                batch_size = labels.size(0)
                for i in range(len(self.models)):
                    input_feat = utils.get_feats(feats, i)
                    prob = self.models[i].test(input_feat)
                    probs_arr[i][index:index+batch_size, :] = prob

                glabels[index:index+batch_size] = labels.data
                index += batch_size

        probs = np.zeros((n_samples, dim))
        for i in range(len(self.models)):
            probs += probs_arr[i] * cfg.MODEL.WEIGHTS[i]
        probs = probs / np.sum(cfg.MODEL.WEIGHTS)

        return probs_arr, probs, glabels

    def eval(self, epoch):
        if epoch == 0:
            return
        self.eval_mode()

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            if (self.distributed == False) or (dist.get_rank() == 0):
                os.mkdir(result_folder)

        #################################### trg ####################################
        probs_arr, probs, labels = self.eval_one(self.trgte_loader)
        mean_acc, preds = utils.eval(epoch, cfg.DATA_LOADER.TARGET, probs, labels)
        #self.logger.info(mean_acc)
        utils.write_list(result_folder, epoch, cfg.DATA_LOADER.TARGET, preds)

        pickle.dump({ 'probs': probs, 'labels': preds }, \
            open(os.path.join(result_folder, str(epoch) + '_fusion_' + cfg.DATA_LOADER.TARGET + '_test.pkl'), 'wb'))

        #for i in range(len(probs_arr)):
        #    mean_acc, preds = utils.eval(epoch, self.names[i] + ' ' + cfg.DATA_LOADER.TARGET, probs_arr[i], labels)
        #    self.logger.info(mean_acc)

        probs_arr, probs, labels = self.eval_one(self.trgte_tr_loader)
        mean_acc, preds = utils.eval(epoch, cfg.DATA_LOADER.TARGET + '_train', probs, labels)
        #self.logger.info(mean_acc)
        utils.write_list(result_folder, epoch, cfg.DATA_LOADER.TARGET + '_train', preds)

        pickle.dump({ 'probs': probs, 'labels': preds }, \
            open(os.path.join(result_folder, str(epoch) + '_fusion_' + cfg.DATA_LOADER.TARGET + '.pkl'), 'wb'))

        #################################### real ####################################
        _, probs, labels = self.eval_one(self.relte_loader)
        mean_acc, preds = utils.eval(epoch, 'real', probs, labels)
        self.logger.info(mean_acc)
        #utils.write_list(result_folder, epoch, 'real', preds)
  
    def eval_mode(self):
        for i in range(len(self.models)):
            self.models[i].eval()

    def train_mode(self):
        for i in range(len(self.models)):
            self.models[i].train(mode=True)

    def zero_grad(self):
        for i in range(len(self.models)):
            self.optims[i].zero_grad()

    def step(self, epoch):
        for i in range(len(self.models)):
            self.optims[i].step(epoch)

    def train(self):
        for epoch in range(0, cfg.SOLVER.MAX_EPOCH):
            self.eval(epoch)
            self.train_mode()

            rel_loader_iter = iter(self.reltr_loader)
            inf_loader_iter = iter(self.inftr_loader)
            qdr_loader_iter = iter(self.qdrtr_loader)
            skt_loader_iter = iter(self.skttr_loader)
            trg_loader_iter = iter(self.trgtr_loader)

            for _ in range(self.iters_epoch):
                rel_feats, rel_labels = rel_loader_iter.next()
                inf_feats, inf_labels = inf_loader_iter.next()
                qdr_feats, qdr_labels = qdr_loader_iter.next()
                skt_feats, skt_labels = skt_loader_iter.next()
                trg_feats, trg_labels = trg_loader_iter.next()
                rel_feats = [feat.cuda() for feat in rel_feats]
                inf_feats = [feat.cuda() for feat in inf_feats]
                qdr_feats = [feat.cuda() for feat in qdr_feats]
                skt_feats = [feat.cuda() for feat in skt_feats]
                rel_labels = rel_labels.cuda()
                inf_labels = inf_labels.cuda()
                qdr_labels = qdr_labels.cuda()
                skt_labels = skt_labels.cuda()
                src_feats = []
                for i in range(len(cfg.MODEL.NET_TYPE)):
                    src_feats.append(torch.cat([rel_feats[i], inf_feats[i], qdr_feats[i], skt_feats[i]], 0))
                src_labels = torch.cat([rel_labels, inf_labels, qdr_labels, skt_labels], 0)

                trg_feats = [feat.cuda() for feat in trg_feats]
                trg_labels = trg_labels.cuda()

                losses = 0
                losses_info = {}
                self.zero_grad()

                for i in range(len(cfg.MODEL.NETS)):
                    src_input = utils.get_feats(src_feats, i)
                    trg_input = utils.get_feats(trg_feats, i)
                    loss, loss_info = self.models[i](src_input, src_labels, trg_input, trg_labels)
                    losses += loss
                    for key in loss_info:
                        losses_info[self.names[i] + ' ' + key] = loss_info[key]
                    
                self.display(self.iteration, losses, losses_info)
                losses.backward()
                self.iteration += 1
                self.step(epoch)
            
            probs_arr, _, _ = self.eval_one(self.trgte_tr_loader)
            self.trgtr_dataset.filter_trg_plabels(probs_arr)
            self.trgtr_loader = data_loader.load_trg_train(self.trgtr_dataset)      
            self.save_model(epoch + 1)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()

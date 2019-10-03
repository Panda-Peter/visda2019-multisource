import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

def eval(epoch, name, test_loader, netG, netE):
    netG.eval()
    netE.eval()

    correct = 0
    tick = 0
    subclasses_correct = np.zeros(cfg.MODEL.CLASS_NUM)
    subclasses_tick = np.zeros(cfg.MODEL.CLASS_NUM)

    results = []
    with torch.no_grad():
        for (_, imgs, labels) in tqdm.tqdm(test_loader):  
            imgs = Variable(imgs.cuda())
            _, _unsup_pool5_out = netG(imgs)
            _, _unsup_logits_out = netE(_unsup_pool5_out)
            pred = F.softmax(_unsup_logits_out, dim=1)
            pred = pred.data.cpu().numpy().argmax(axis=1)
            results += list(pred)

            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    correct += 1
                    subclasses_correct[pred[i]] += 1
                tick += 1
                
            if epoch == 0:
                return None, None

    correct = correct * 1.0 / tick
    subclasses_result = np.divide(subclasses_correct, subclasses_tick)
    mean_class_acc = subclasses_result.mean()
    zeors_num = subclasses_result[subclasses_result == 0].shape[0]

    mean_acc_str = "*** Epoch {:d}, {:s}; mean class acc = {:.2%}, overall = {:.2%}, missing = {:d}".format(\
        epoch, name, mean_class_acc, correct, zeors_num)
    
    return mean_acc_str, results

def evalfeat(test_loader, netG, netE):
    netG.eval()
    netE.eval()

    dim = cfg.MODEL.EMBED_DIM
    n_samples = test_loader.dataset.__len__()
    feats = torch.zeros(n_samples, dim).cuda()
    labels = torch.zeros(n_samples).cuda()
    probs = torch.zeros(n_samples, cfg.MODEL.CLASS_NUM).cuda()

    with torch.no_grad():
        index = 0
        for _, imgs, label in tqdm.tqdm(test_loader):
            batch_size = imgs.size(0)
            imgs = Variable(imgs.cuda())
            label = Variable(label.cuda())
            _, _unsup_pool5_out = netG(imgs)
            features, _unsup_logits_out = netE(_unsup_pool5_out)
            prob = F.softmax(_unsup_logits_out, dim=1)

            feats[index:index+batch_size, :] = features.data
            labels[index:index+batch_size] = label.data
            probs[index:index+batch_size] = prob.data
            index += batch_size
    return probs, feats, labels

def evalprobs(test_loader, netG, netE):
    netG.eval()
    netE.eval()

    dim = cfg.MODEL.CLASS_NUM
    n_samples = test_loader.dataset.__len__()
    probs = torch.zeros(n_samples, dim).cuda()
    plabels = np.zeros((n_samples,), dtype='int')

    with torch.no_grad():
        index = 0
        for _, imgs, _ in tqdm.tqdm(test_loader):
            batch_size = imgs.size(0)
            imgs = Variable(imgs.cuda())
            _, _unsup_pool5_out = netG(imgs)
            _, _unsup_logits_out = netE(_unsup_pool5_out)
            prob = F.softmax(_unsup_logits_out, dim=1)
            plabel = prob.data.cpu().numpy().argmax(axis=1)

            probs[index:index+batch_size, :] = prob.data
            plabels[index:index+batch_size] = plabel.data 
            index += batch_size
    
    probs = probs.data.cpu().numpy()
    return probs, plabels
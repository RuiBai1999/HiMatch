#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from train_modules.evaluation_metrics import evaluate
import torch
import tqdm
import random
import numpy as np
import os
#for debug
#torch.autograd.set_detect_anomaly(True)

class Trainer(object):
    def __init__(self, model, criterion, optimizer, vocab, config, mode="TRAIN"):
        """
        :param model: Computational Graph
        :param criterion: train_modules.ClassificationLoss object
        :param optimizer: optimization function for backward pass
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        if mode=="TRAIN":
            self.criterion, self.criterion_ranking = criterion[0], criterion[1]
        else:
            self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = config.data.dataset

    def update_lr(self):
        """
        (callback function) update learning rate according to the decay weight
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run(self, data_loader, epoch, mode='TRAIN', label_desc_loader=None):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :param label_desc_loader: Dataloader of label description
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels = []
        total_loss = 0.0

        label_repre = -1
        if mode == "TRAIN":
            for batch_i, batch in enumerate(label_desc_loader):
                label_embedding = self.model.get_embedding([batch, mode])
                if batch_i == 0:
                    label_repre = label_embedding
                else:
                    label_repre = torch.cat([label_repre, label_embedding], 0)
        for batch in tqdm.tqdm(data_loader):
            if mode != "TRAIN":
                logits = self.model([batch, mode, label_repre])
            else:
                logits, text_repre, label_repre_positive, label_repre_negative = self.model([batch, mode, label_repre])
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None
            loss = self.criterion(logits,
                                  batch['label'].to(self.config.train.device_setting.device),
                                  recursive_constrained_params)
            total_loss += loss.item()

            if mode == 'TRAIN':
                loss_inter, loss_intra = self.criterion_ranking(text_repre, label_repre_positive, label_repre_negative, batch['margin_mask'].to(self.config.train.device_setting.device))

                self.optimizer.zero_grad()
                loss = loss + loss_inter + loss_intra
                loss.backward(retain_graph=True)
                self.optimizer.step()

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])
        if mode == 'TRAIN':
            logger.info("loss: %f" % (loss.item()))
        elif mode == 'EVAL':
            metrics = evaluate(predict_probs,
                               target_labels,
                               self.vocab,
                               self.config.eval.threshold)
            logger.info("Performance at epoch %d --- Precision: %f, "
                        "Recall: %f, Micro-F1: %f, Macro-F1: %f"
                        % (epoch, metrics['precision'], metrics['recall'], metrics['micro_f1'], metrics['macro_f1']))
            return metrics


    def train(self, data_loader, label_desc_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, mode='TRAIN', label_desc_loader=label_desc_loader)

    def eval(self, data_loader, epoch, mode='EVAL'):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, mode=mode)

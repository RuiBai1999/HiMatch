import torch.nn as nn
import math
import torch
import numpy as np
import os
from models.structure_model.structure_encoder import StructureEncoder

class MatchingNet(nn.Module):
    def __init__(self, config, graph_model_label=None, label_map=None):
        super(MatchingNet, self).__init__()
        self.dataset = config.data.dataset
        self.label_map = label_map
        self.positive_sample_num = config.data.positive_num
        self.negative_sample_num = config.data.negative_ratio * self.positive_sample_num

        self.embedding_net1 = nn.Sequential(nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension, 200),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Linear(200, 200),
                                                nn.ReLU(),
                                                nn.Dropout(0.5))
        self.label_encoder = nn.Sequential(nn.Linear(config.embedding.label.dimension, 200),
                                               nn.ReLU(),
                                               nn.Dropout(0.5),
                                               nn.Linear(200, 200),
                                               nn.ReLU(),
                                               nn.Dropout(0.5))

        self.graph_model = graph_model_label
        self.label_map = label_map

    def forward(self, text, gather_positive, gather_negative, label_repre):
        """
        forward pass of matching learning
        :param gather_positive ->  torch.BoolTensor, (batch_size, positive_sample_num), index of positive label
        :param gather_negative ->  torch.BoolTensor, (batch_size, negative_sample_num), index of negative label
        :param label_repre ->  torch.FloatTensor, (batch_size, label_size, label_feature_dim)
        """
        gather_positive = gather_positive.to(text.device)
        gather_negative = gather_negative.to(text.device)

        label_repre = label_repre.unsqueeze(0)
        label_repre = self.graph_model(label_repre)
        label_repre = label_repre.repeat(text.size(0), 1, 1)

        label_repre = self.label_encoder(label_repre)
        label_positive = torch.gather(label_repre, 1, gather_positive.view(text.size(0), self.positive_sample_num, 1).expand(text.size(0), self.positive_sample_num, label_repre.size(-1)))

        label_negative = torch.gather(label_repre, 1, gather_negative.view(text.size(0), self.negative_sample_num, 1).expand(text.size(0), self.negative_sample_num, label_repre.size(-1)))
        text_encoder = self.embedding_net1(text)

        return text_encoder, label_positive, label_negative


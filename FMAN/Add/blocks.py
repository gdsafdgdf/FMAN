import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self,units_in, units_out, activation, kernel_initializer='glorot_uniform', batchnorm=False, normalizer="batch", trainable=True, name="mlpblock"):
        super(MLPBlock, self).__init__()

        self.act = activation
        self.fc = nn.Linear(units_in, units_out)
        self.out_features = units_out

        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.fc.weight)

        if not trainable:
            for param in self.fc.parameters():
                param.requires_grad = False

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.normalizer = nn.BatchNorm1d(units_out) if normalizer == "batch" else nn.LayerNorm(units_out)

    def forward(self, inputs, training = False):
        features = self.fc(inputs)

        if self.batchnorm:
            features = self.normalizer(features)
        features = self.act(features)

        return features

class DensenetBlock(nn.Module):
    def __init__(self, units_in, units_out, activation, kernel_initializer='glorot_uniform', batchnorm=False, normalizer="batch", trainable=True, name="denseblock"):
        super(DensenetBlock, self).__init__()

        self.act = activation
        self.fc = nn.Linear(units_in, units_out)
        self.out_features = units_out + units_in

        # Apply He initialization
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')

        if not trainable:
            for param in self.fc.parameters():
                param.requires_grad = False

        self.batchnorm = batchnorm
        self.normalizer = normalizer
        if batchnorm:
            self.BatchNorm1d = nn.BatchNorm1d(units_out)
            self.LayerNorm = nn.LayerNorm(units_out, eps=1e-5)

    def forward(self, inputs):
        identity_map = inputs

        features = self.fc(inputs)

        if self.batchnorm:
            if self.normalizer == 'batch':
                features = self.BatchNorm1d(features)
            elif self.normalizer == 'layer':
                features = self.LayerNorm(features)

        features = self.act(features)
        features = torch.cat([features, identity_map], dim=1)

        return features
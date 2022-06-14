#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 下午5:41
# @Author  : xiaot
from collections import OrderedDict

import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from pytorch_util.conf.activation_name_conf import *
from pytorch_util.conf.mlp_model_conf import IS_USING_CUDA
from pytorch_util.conf.optimizer_name_conf import *
from pytorch_util.src.util.gpu_getter import try_gpu
from pytorch_util.src.util.io_util.logging_instance import logger


class MLPModel:

    def __init__(self,
                 list_n_features_per_layer,
                 list_prob_dropout=None,
                 activation_function=ACTIVATION_NAME_RELU):
        """
        get a mlp model with the given list_n_features_per_layer setting
        :param list_n_features_per_layer:
            example: [784, 50, 50, 10] means:
                input layer: 784 nodes
                hidden layer 1: 50 nodes
                hidden layer 2: 50 nodes
                output layer: 10 nodes
        :param list_prob_dropout:
            example: [0.5, 0.5]
        :param activation_function:
            'sigmoid', 'tanh', 'relu'
        :return: a mlp model with the given list_n_features_per_layer setting
        """
        assert not list_prob_dropout or len(
            list_n_features_per_layer) == len(list_prob_dropout) + 2

        n_layers_net = len(
            list_n_features_per_layer) - 1  # input layer doesn't count

        lst_model = []
        for l in range(1, n_layers_net + 1):
            # add linear layer
            layer_name_linear = 'linear' + str(l)
            in_features = list_n_features_per_layer[l - 1]
            out_features = list_n_features_per_layer[l]
            tuple_layer_linear_now = (layer_name_linear,
                                      nn.Linear(in_features=in_features,
                                                out_features=out_features))
            lst_model.append(tuple_layer_linear_now)

            if l < n_layers_net:
                # add dropout layer
                if list_prob_dropout:
                    layer_name_dropout_now = 'dropout' + str(l)
                    prob_dropout_now = list_prob_dropout[l - 1]
                    layer_dropout_now = nn.Dropout(prob_dropout_now)
                    tuple_layer_dropout_now = (layer_name_dropout_now,
                                               layer_dropout_now)
                    lst_model.append(tuple_layer_dropout_now)

                # add activation layer
                layer_name_activation = activation_function + str(l)
                if activation_function == ACTIVATION_NAME_SIGMOID:
                    layer_activation_now = nn.Sigmoid()
                elif activation_function == ACTIVATION_NAME_TANH:
                    layer_activation_now = nn.Tanh()
                elif activation_function == ACTIVATION_NAME_RELU:
                    layer_activation_now = nn.ReLU()
                else:
                    assert activation_function in SET_ACTIVATION_NAME
                tuple_layer_activation_now = (layer_name_activation,
                                              layer_activation_now)
                lst_model.append(tuple_layer_activation_now)

                # net = nn.Sequential()
                # for i in range(4):
                #     net.add_module(f'block {i}', block1())

        od = OrderedDict(lst_model)
        model = nn.Sequential(od)
        logger.info('model.parameters: %s' % model.parameters)

        model.eval()

        self._model = model

        # try gpu
        self._device = try_gpu(0)
        if IS_USING_CUDA:
            self._model = self._model.to(self._device)

        self._list_n_features_per_layer = list_n_features_per_layer

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def list_n_features_per_layer(self):
        return self._list_n_features_per_layer

    def get_optimizer(self, optimizer_name, lr):
        if optimizer_name == OPTIMIZER_NAME_SGD:
            optimizer = optim.SGD(self._model.parameters(), lr=lr)
        elif optimizer_name == OPTIMIZER_NAME_ADAM:
            optimizer = optim.Adam(self._model.parameters(), lr=lr)
        else:
            assert optimizer_name in SET_OPTIMIZER_NAME
        return optimizer

    def train(self,
              loader_train,
              loader_validation,
              len_train,
              len_validation,
              criterion,
              optimizer,
              n_epochs=100):
        list_loss_train = []
        list_accuracy_train = []
        list_accuracy_validation = []
        self._model.train()
        for epoch in range(n_epochs):
            loss_this_epoch = 0
            n_correct_train = 0
            for x, y in loader_train:

                # accelerate using gpu
                if IS_USING_CUDA:
                    x, y = x.to(self._device), y.to(self._device)

                # eval mode
                self._model.eval()

                y_pred = self._model(x.view(-1, self._list_n_features_per_layer[0]))

                # train mode
                self._model.train()

                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # cumulative loss
                loss_this_epoch += loss.item()

                max_probs, y_pred = torch.max(y_pred, 1)
                n_correct_train += (y_pred == y).sum().item()

                # PlotStuff(dataset.x, dataset.y, self._model)

            accuracy_train = n_correct_train / len_train
            list_accuracy_train.append(accuracy_train)

            logger.info('epoch: %d, training loss: %f' %
                        (epoch + 1, loss_this_epoch))
            logger.info('epoch: %d, training accuracy: %f' %
                        (epoch + 1, accuracy_train))

            list_loss_train.append(loss_this_epoch)

            # eval mode
            self._model.eval()

            n_correct = 0
            for x, y in loader_validation:

                # accelerate using gpu
                if IS_USING_CUDA:
                    x, y = x.to(self._device), y.to(self._device)


                y_pred = self._model(x.view(-1, self._list_n_features_per_layer[0]))

                max_probs, y_pred = torch.max(y_pred, 1)
                n_correct += (y_pred == y).sum().item()

            accuracy = n_correct / len_validation
            list_accuracy_validation.append(accuracy)
            logger.info('epoch: %d, validation accuracy: %f' %
                        (epoch + 1, accuracy))

            # train mode
            self._model.train()

        self._model.eval()

        return list_loss_train, list_accuracy_train, list_accuracy_validation

    @staticmethod
    def plot_train(nepochs, list_loss_train, list_accuracy_train,
                   list_accuracy_validation):
        # plot loss
        plt.plot(range(1, nepochs + 1), list_loss_train, label='training loss')
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        plt.legend()
        plt.show()

        # plt train & validation accuracy
        plt.plot(range(1, nepochs + 1),
                 list_accuracy_train,
                 label='training accuracy')
        plt.plot(range(1, nepochs + 1),
                 list_accuracy_validation,
                 label='validation accuracy')
        plt.xlabel('epochs')
        plt.ylabel('training/validation accuracy')
        plt.legend(['training accuracy', 'validation accuracy'])
        plt.show()

    def __call__(self, *args, **kwargs):
        return self._model.__call__(*args, **kwargs)

    def forward(self, x):
        """
        forward / predict
        :param x:
        :return:
        """
        if IS_USING_CUDA:
            x = x.to(self._device)
        return self._model.forward(x)

    def predict(self, X):
        if type(X) == list:
            X = [[float(f) for f in x] for x in X]
            X = torch.tensor(X)
        if X.device != self._device:
            X = X.to(self._device)

        Y_pred = self._model(X.view(-1, self._list_n_features_per_layer[0]))
        max_probs, Y_pred = torch.max(Y_pred, 1)

        return Y_pred

    def test(self, loader_test, len_dataset_test):
        n_correct = 0
        for x, y in loader_test:

            # accelerate using gpu
            if IS_USING_CUDA:
                x, y = x.to(self._device), y.to(self._device)

            y_pred_test = self._model(x.view(-1, self._list_n_features_per_layer[0]))

            max_probs, y_pred_test = torch.max(y_pred_test, 1)
            n_correct += (y_pred_test == y).sum().item()

        accuracy = n_correct / len_dataset_test
        return accuracy

    def save_state_dict(self, file_path):
        """
        save model parameters
        :param file_path:
        :return:
        """
        torch.save(self._model.state_dict(), file_path)

    def load_state_dict(self, file_path):
        """
        load model parameters
        :param file_path:
        :return:
        """
        self._model.load_state_dict(torch.load(file_path))

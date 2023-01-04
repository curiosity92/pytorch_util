#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 下午1:58
# @Author  : xiaot
from random import randint

import torch
from torch import nn, manual_seed
manual_seed(1)  # TODO set pytorch seed

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from pytorch_util.src.util.io_util.logging_instance import logger
from pytorch_util.conf.dataset_conf import RATIO_TRAINING_SET, RATIO_VALIDATION_SET
from pytorch_util.conf.base_model_conf import IS_USING_CUDA
from pytorch_util.conf.base_model_conf import MODEL_FILE_PATH
from pytorch_util.conf.mlp_model_conf import LIST_N_FEATURES_PER_LAYER, LIST_PROB_DROPOUT, ACTIVATION_FUNCTION_HIDDEN
from pytorch_util.conf.train_conf import BATCH_SIZE_TRAIN, BATCH_SIZE_VALIDATION, N_EPOCHS, LEARNING_RATE, \
    OPTIMIZER_NAME, BATCH_SIZE_TEST
from pytorch_util.src.util.dataset_util import TorchData
from pytorch_util.src.util.random_shuffler import shuffle_2_lists
from pytorch_util.src.util.dataset_csv_reader import get_X_Y_from_csv
from pytorch_util.src.util.mlp_model_agent import MLPModelAgent

# TODO change here
IS_LOAD_MODEL = False
IS_TRAIN = True


class TrainData1(Dataset):

    def __init__(self):
        self.X = torch.arange(0, 5000 * 784, 1.0).view(-1, 28, 28)  # 5000 * 1 * 28 * 28
        # self.Y = torch.tensor([i // (5000 / 10) for i in range(5000)], dtype=torch.int64).view(-1)  # 5000
        # self.Y = torch.tensor([i % 10 for i in range(5000)]).view(-1)
        self.Y = torch.tensor([5 + randint(-1, 1) for i in range(5000)]).view(-1)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


class ValidationData1(Dataset):

    def __init__(self):
        self.X = torch.arange(1000, 1000 + 5000 * 784, 1.0).view(-1, 28, 28)  # 5000 * 1 * 28 * 28
        self.Y = torch.tensor([(1 + i // (5000 / 10)) % 10 for i in range(5000)], dtype=torch.int64).view(-1)  # 5000
        # self.Y = torch.tensor([i % 10 for i in range(5000)]).view(-1)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def get_datasets_0():
    dataset_train = dsets.MNIST(root='../data',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())
    dataset_validation = dsets.MNIST(root='../data',
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())
    dataset_test = dataset_validation
    return dataset_train, dataset_validation, dataset_test


def get_datasets_1():
    dataset_train = TrainData1()
    dataset_validation = ValidationData1()
    dataset_test = dataset_validation
    return dataset_train, dataset_validation, dataset_test


def get_datasets_2():
    index_x = 3
    index_y = 4

    # positive samples
    file_name = 'positive_samples.csv'
    all_positive_X, all_positive_Y = get_X_Y_from_csv(file_name, index_x, index_y)
    logger.debug('read positive samples')

    # negative samples
    file_name = 'negative_samples.csv'
    all_negative_X, all_negative_Y = get_X_Y_from_csv(file_name, index_x, index_y)
    logger.debug('negative samples read')

    # all samples
    all_X, all_Y = all_positive_X + all_negative_X, all_positive_Y + all_negative_Y
    logger.debug('all samples read')

    shuffled_X, shuffled_Y = shuffle_2_lists(list1=all_X, list2=all_Y)
    logger.debug('all samples shuffled')

    train_X = shuffled_X[0: int(len(shuffled_X) * RATIO_TRAINING_SET)]
    train_Y = shuffled_Y[0: int(len(shuffled_Y) * RATIO_TRAINING_SET)]
    validation_X = shuffled_X[len(train_X): len(train_X) + int(len(shuffled_X) * RATIO_VALIDATION_SET)]
    validation_Y = shuffled_Y[len(train_Y): len(train_Y) + int(len(shuffled_Y) * RATIO_VALIDATION_SET)]
    test_X = shuffled_X[len(train_X) + len(validation_X):]
    test_Y = shuffled_Y[len(train_Y) + len(validation_Y):]

    dataset_train = TorchData(X=train_X, Y=train_Y,  input_dim=LIST_N_FEATURES_PER_LAYER[0])
    dataset_validation = TorchData(X=validation_X, Y=validation_Y, input_dim=LIST_N_FEATURES_PER_LAYER[0])
    dataset_test = TorchData(X=test_X, Y=test_Y, input_dim=LIST_N_FEATURES_PER_LAYER[0])

    return dataset_train, dataset_validation, dataset_test


# get train & validation dataset
# dataset_train, dataset_validation, dataset_test = get_datasets_0()
# dataset_train, dataset_validation, dataset_test = get_datasets_1()
dataset_train, dataset_validation, dataset_test = get_datasets_2()
logger.debug('dataset built')
# logger.debug('dataset_train[0]:', dataset_train[0])

# get train & validation & test loader
loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE_TRAIN)
loader_validation = DataLoader(dataset=dataset_validation,
                               batch_size=BATCH_SIZE_VALIDATION)
loader_test = DataLoader(dataset=dataset_test,
                         batch_size=BATCH_SIZE_TEST)
logger.debug('dataloader built')


# build model
model, device = MLPModelAgent.build_model(list_n_features_per_layer=LIST_N_FEATURES_PER_LAYER,
                      list_prob_dropout=LIST_PROB_DROPOUT,
                      activation_function=ACTIVATION_FUNCTION_HIDDEN)
model_agent = MLPModelAgent(model=model, device=device, list_n_features_per_layer=LIST_N_FEATURES_PER_LAYER)
logger.info('model initialized')

# load model
if IS_LOAD_MODEL:
    model_agent.load_state_dict(file_path=MODEL_FILE_PATH)
    logger.info('model weights loaded')


def train():
    # loss
    loss = nn.CrossEntropyLoss()
    logger.debug('loss defined')

    # optimizer
    optimizer = model_agent.get_optimizer(optimizer_name=OPTIMIZER_NAME, lr=LEARNING_RATE)
    logger.debug('optimizer defined')

    # train model
    list_loss_train, list_accuracy_train, list_accuracy_validation = model_agent.train(
        iter_train=loader_train,
        iter_validation=loader_validation,
        len_train=len(dataset_train),
        len_validation=len(dataset_validation),
        criterion=loss,
        optimizer=optimizer,
        n_epochs=N_EPOCHS)
    logger.info('model training completed')

    # plot train
    MLPModelAgent.plot_train(n_epochs=N_EPOCHS,
                             list_loss_train=list_loss_train,
                             list_accuracy_train=list_accuracy_train,
                             list_accuracy_validation=list_accuracy_validation)
    logger.debug('graph plotted')

    # save model
    model_agent.save_state_dict(file_path=MODEL_FILE_PATH)
    logger.info('model saved')


# train model
if IS_TRAIN:
    train()

# dropout 0, SGD, lr=0.01
# epoch: 1, training loss: 68.892220
# epoch: 1, validation accuracy: 0.167000
# epoch: 2, training loss: 68.649961
# epoch: 2, validation accuracy: 0.195200
# epoch: 3, training loss: 68.396866
# epoch: 3, validation accuracy: 0.236300
# epoch: 4, training loss: 68.118833
# epoch: 4, validation accuracy: 0.272900
# epoch: 5, training loss: 67.803950
# epoch: 5, validation accuracy: 0.298400
# ...
# epoch: 35, training loss: 23.709590
# epoch: 35, validation accuracy: 0.810900

# dropout 0.5, SGD, lr=0.01
# epoch: 35, training loss: 47.561431
# epoch: 35, validation accuracy: 0.704600

# dropout 1.0, SGD, lr=0.01
# epoch: 35, training loss: 69.040879
# epoch: 35, validation accuracy: 0.152200

# dropout 0, ADAM, lr=0.01
# epoch: 1, training loss: 23.251255
# epoch: 1, training accuracy: 0.769300
# epoch: 1, validation accuracy: 0.904400
# epoch: 2, training loss: 8.518528
# epoch: 2, training accuracy: 0.918200
# epoch: 2, validation accuracy: 0.935300
# epoch: 3, training loss: 6.386681
# epoch: 3, training accuracy: 0.938033
# epoch: 3, validation accuracy: 0.946800
# epoch: 4, training loss: 5.188486
# epoch: 4, training accuracy: 0.949517
# epoch: 4, validation accuracy: 0.952700
# epoch: 5, training loss: 4.323868
# epoch: 5, training accuracy: 0.957717
# epoch: 5, validation accuracy: 0.956600
# ...
# epoch: 35, training loss: 0.677103
# epoch: 35, training accuracy: 0.992300
# epoch: 35, validation accuracy: 0.968800


# test model_agent
accuracy = model_agent.test(iter_test=loader_test, len_dataset_test=len(dataset_test))
print('test accuracy: %f' % accuracy)

# predict
X = [[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
Y_pred = model_agent.predict(X=X)
print('Y_pred: %s' % Y_pred)
print('Y_pred == torch.tensor([1, 1, 0], device=model.device): %s' % (Y_pred == torch.tensor([1, 1, 0], device=model_agent.device)))
assert Y_pred.tolist() == [1, 1, 0]

import code, os, shutil, tflearn, datetime, sys, git, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
import numpy as np
from utils import *
from Model import *
from Dataloader import Dataloader

class Trainer:
    def __init__(self, config):
        # create training necessasities
        self.dataloader = Dataloader(self.config)
        #
        self.model = DartsTransformer(config).cuda()
        #
        self.loss_fn = torch.nn.CrossEntropyLoss # TODO sequence masking
        #
        self.optimizer = torch.optim.Adam # TODO set the learning rate?
        # initialises all variables
        self.initialize_variables()
        #
        # self.config['num_weights'] = count_train_vars(self) # TODO torch style
        self.begin = time.time()

    # TODO write wrapper function, that creates pytorch loss
    def create_loss(self):
        if self.config['embedding_type'] == 'word2vec':
            self.train_loss = tf.losses.mean_squared_error(self.train_prediction, self.groundtruth)
        elif self.config['embedding_type'] == 'one_hot':
            prediction_length = tf.argmax(tf.argmax(self.train_prediction, -1), -1)
            groundtruth_length = tf.argmax(self.groundtruth, -1)
            seq_length = tf.maximum(prediction_length, groundtruth_length)
            weights = tf.sequence_mask(seq_length, self.config['max_sentence_len'], tf.float32)
            weights = tf.cast(weights, tf.float32)
            weights = tf.ones([self.train_prediction.shape[0], self.config['max_sentence_len']])
            self.train_loss = tf.contrib.seq2seq.sequence_loss(self.train_prediction, self.groundtruth, weights)

    # TODO does this include functionality pytorch doesn't provide?
    def create_optimizer(self, reuse=None):
        with tf.variable_scope("optimizer", reuse=reuse):
            optimizer = tf.train.MomentumOptimizer(self.config['current_learning_rate'],0.9)
            self.weights = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, self.weights)
            self.clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2.0) # in [1,5]
            self.update_step = optimizer.apply_gradients(zip(self.clipped_gradients, self.weights))
            l = []
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer'):
                l.append(v)
            init = tf.variables_initializer(l)
            self.session.run(init)

    # TODO did i forgot something?
    def initialize_variables(self):
        if not self.config['is_loaded']:
            for m in self.model.modules():
                m.weight.data.normal_(0, self.config['initialization'])
            #save_current(self) #TODO!
            #save_seeds(self)
            #save_best(self)
            '''init = tf.global_variables_initializer()
            self.session.run(init)
            for v in tf.trainable_variables():
                seed = np.random.standard_normal(v.shape.as_list()) * self.config['initialization']
                v = v.assign(seed)
                x = self.session.run(v)
                #print(seed.shape)'''
        else:
            restore_current(self)

    # TODO how to implement?
    def decay_learning_rate(self):
        self.config['current_learning_rate'] *= self.config['learning_rate_decay']
        self.create_optimizer(reuse=True)

    def train_step(self, src, tgt):
        '''
        Propagates one Minibatch through the network and updates the weights accordingly.
        inp : B x H x W x 1
        groundtruth: B x L
        '''
        self.model.train()
        pred, act = self.model(src.cuda(), tgt)
        loss = self.loss_fn(pred, tgt)
        # manually zero all previous gradients
        self.optimizer.zero_grad()
        # calculate new gradients
        loss.backward()
        # apply new gradients
        self.optimizer.step()
        #
        pred_cpu = pred.cpu()
        loss_cpu = loss.cpu()
        # calculate accuracies
        self.current_pred = get_batch_tokens(self, pred_cpu)
        self.current_src = get_batch_tokens(self, src, 10)
        self.current_tgt = get_batch_tokens(self, tgt, 10)
        #
        self.current_accuracy = calculate_batch_accuracy(self, self.current_pred, self.current_tgt)
        self.current_bleu = calculate_batch_accuracy(self, self.current_tgt, self.current_tgt, mode='eda')
        #
        self.losses.append(self.current_loss)
        self.accuracies.append(self.current_accuracy)
        self.bleus.append(self.current_bleu)
        #
        self.end = time.time()
        self.config['current_time'] += (self.end - self.begin)
        self.begin = time.time()
        # increase current step
        if self.config['current_step'] % (self.config['log_freq'] * 5) == 0:
            # print_gradients(self, self.current_gradients, self.current_weights) # TODO extract gradients and weights
        if self.config['current_step'] % self.config['log_freq'] == 0:
            print_logs(self, self.current_tgt, self.current_src, 'train')
        if self.config['current_step'] % 1000 == 0 and self.config['current_step'] != 0:
            print(self.config['current_step'])
            io_sucessful = False
            while not io_sucessful:
                try:
                    save_current(self)
                    io_sucessful = True
                except Exception as e:
                    print('problems with I/O')
                    print(e)
                    time.sleep(5)
        self.config['current_step'] += 1

    def val_step(self, src_embedded, groundtruth_embedded):
        '''
        Propagates one Minibatch through the network without updating the weights.
        inp : B x H x W x 1
        groundtruth: B x L
        '''
        self.model.eval()
        pred, act = self.model(src.cuda(), dropout=0.0)
        loss = self.loss_fn(pred, tgt)
        #
        pred_cpu = pred.cpu()
        loss_cpu = loss.cpu()
        # calculate accuracies
        self.current_pred = get_batch_tokens(self, pred_cpu)
        self.current_src = get_batch_tokens(self, src, 10)
        self.current_tgt = get_batch_tokens(self, tgt, 10)
        #
        self.current_accuracy = calculate_batch_accuracy(self, self.current_pred, self.current_tgt)
        self.current_bleu = calculate_batch_accuracy(self, self.current_tgt, self.current_tgt, mode='eda')
        #
        self.losses.append(self.current_loss)
        self.accuracies.append(self.current_accuracy)
        self.bleus.append(self.current_bleu)
        #
        self.end = time.time()
        self.config['current_time'] += (self.end - self.begin)
        self.begin = time.time()
        # increase current step
        print_logs(self, self.current_groundtruth, self.current_source, 'val')

    def run_epoch(self, phase):
        self.losses = []
        self.accuracies = []
        self.bleus = []
        if phase == 'train':
            self.dataloader.shuffle_train_data()
        #
        current_batch, is_finished = self.dataloader.next_batch(phase)
        while not is_finished:
            if phase == 'train':
                self.train_step(current_batch['train'][0]['src'], current_batch['train'][0]['tgt_vecs'])
            elif phase == 'val':
                self.val_step(current_batch['train'][0]['src'], current_batch['train'][0]['tgt_vecs'])
            elif phase == 'test':
                self.test_step(current_batch['train'][0]['src'], current_batch['train'][0]['tgt_vecs'])
            current_batch, is_finished = self.dataloader.next_batch(phase)
        return float(np.mean(self.losses)), float(np.mean(self.accuracies)), float(np.mean(self.bleus))
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time, git
import argparse
import os, code
import json
from utils import *
from ModelFactory import Model
#code.interact(local=dict(globals(), **locals()))


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load_from", default=None, help="specify path, where config of loaded model lies")
parser.add_argument("-s", "--skip_preprocessing", default=True, help="specify path, where config of loaded model lies")
parser.add_argument("-t", "--test", default=False, help="specify path, where config of loaded model lies")
parser.add_argument("-c", "--commit", default=False, help="specify path, where config of loaded model lies")
args = parser.parse_args()

# blub
"""Holds model hyperparams and data information."""
base_config = {
    # common paths
    'base_model_dir' : '/cvhci/data/docs/math_expr/printed/im2latex-100k/simple_chitchat/mt_models',
    'dataset_path' : '/cvhci/data/docs/math_expr/printed/im2latex-100k/simple_chitchat/data/imsdb.json',
    'vocab_path' : '/cvhci/data/docs/math_expr/printed/im2latex-100k/simple_chitchat/data/imsdb_vocab_aprox.lst',
    #'vocab_path' : '/cvhci/data/docs/math_expr/printed/im2latex-100k/gan_chitchat/data/GoogleNews-vectors-negative300.bin',
    'log_dir' : None,
    'seed_weights_path' : None,
    'current_weights_path' : None,
    'best_weights_path' : None,
    'log_freq' : 1,
    # parameters that track current status of training
    'current_val_accuarcy' : 0.0,
    'best_val_accuracy' : 0.0,
    'current_val_eda' : 0.0,
    'best_val_eda' : 0.0,
    'current_val_loss' : float('inf'),
    'best_val_loss' : float('inf'),
    'is_loaded' : False,
    'current_step' : 0,
    'current_epoch' : 0,
    'current_time' : 0.0,
    'num_weights' : None,
    # common parameters for nearly all neural networks
    'batch_size' : 100,
    'max_epochs' : 12,
    'initial_learning_rate' : 0.1,
    'current_learning_rate' : 0.1,
    'min_learning_rate' : 0.000001,
    'learning_rate_decay' : 0.5,
    'dropout' : 0.9,
    # model specific parameters
    'embedding_type': 'one_hot',
    'vocab_size' : 20000,
    'attention_size' : 50,
    'embedding_size' : 300,
    'hidden_size' : 300,
    'eos_threshhold' : 0.5,
    'same_threshhold' : 0.9,
    'max_sentence_len' : 50,
    'initialization':0.05,
    # pytorch model parameters
    "num_vocab" : 20000, 
    "embedding_size" : 300, 
    "hidden_size" : 300, 
    "num_layers" : 8, 
    "num_heads" : 4, 
    "total_key_depth" : 300, 
    "total_value_depth" : 300,
    "output_depth" : 300,
    "filter_size" : 300, 
    "max_length" : 71, 
    "input_dropout" : 0.0, 
    "layer_dropout" : 0.0, 
    "attention_dropout" : 0.0, 
    "relu_dropout" : 0.0, 
    "use_mask" : False, 
    "act" : False
}

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
# setup config
if args.load_from == None:
    config = base_config
    config['is_commited'] = args.commit
    config['model_name'] = str(time.strftime("%Y-%m-%d %H %M %S").replace(' ','_').replace('-','_'))
    config['skip_preprocessing'] = 'True' == str(args.skip_preprocessing)
    if config['is_commited']:
        config['model_name'] = 'com' + config['model_name']
    os.makedirs(config['base_model_dir'] + '/' + config['model_name'])
    config['log_dir'] = config['base_model_dir'] + '/' + config['model_name'] + '/logs'
    os.makedirs(config['log_dir'])
    config['weights_dir'] = config['base_model_dir'] + '/' + config['model_name'] + '/weights'
    os.makedirs(config['weights_dir'])
    config['current_weights_path'] = config['weights_dir'] + '/current.weights'
    config['best_weights_path'] = config['weights_dir'] + '/best.weights'
    config['seed_weights_path'] = config['weights_dir'] + '/seed.weights'
    #
    config['train_accuracy_log_path'] = config['log_dir'] + '/train_accuracy_log.txt'
    config['train_loss_log_path'] = config['log_dir'] + '/train_loss_log.txt'
    config['val_accuracy_log_path'] = config['log_dir'] + '/val_accuracy_log.txt'
    config['val_loss_log_path'] = config['log_dir'] + '/val_loss_log.txt'
    config['config_path'] = config['base_model_dir'] + '/' + config['model_name'] + '/config.json'
    if config['is_commited']:
        os.system('git add *.py')
        os.system('git commit -m "model ' + config['model_name'] + ' started!"')
    repo = git.Repo(search_parent_directories=True)
    new_sha = repo.head.object.hexsha
    config['git_hash'] = new_sha
    write_config(config)
else:
    config_path = args.load_from
    if len(args.load_from.split('/')) == 1:
        config_path = base_config['base_model_dir'] + '/' + config_path
    if config_path[len('/config.json'):] != '/config.json':
        config_path += '/config.json'
    config_reader = open(config_path,'r')
    config_string = config_reader.read()
    config_reader.close()
    config_string = config_string.replace(',\n',',').replace('{\n','{').replace('\n}','}')
    config = json.loads(config_string)
    if config['is_commited']:
        os.system('git add *.py')
        os.system('git commit -m "model ' + config['model_name'] + ' restarted!"')
        if config['git_hash'] != sha:
            print('git hash doesnt match!')
            code.interact(local=dict(globals(), **locals()))

model = Model(config)

if config['is_loaded']:
    save_current(model)
else:
    config['is_loaded'] = True

print('==> starting training')
while config['current_epoch'] < config['max_epochs']:
    #
    train_loss, train_accuracy, train_eda = model.run_epoch('train')
    write_log(config['train_loss_log_path'], train_loss)
    write_log(config['train_accuracy_log_path'], train_accuracy)
    #
    model.config['current_val_loss'], model.config['current_val_accuracy'], model.config['current_val_eda'] = model.run_epoch('val')
    write_log(config['val_loss_log_path'], model.config['current_val_loss'])
    write_log(config['val_accuracy_log_path'], model.config['current_val_accuracy'])

    config['current_epoch'] += 1
    model.decay_learning_rate()

    if model.config['current_val_loss'] < model.config['best_val_loss']:
        model.config['best_val_loss'] = model.config['current_val_loss']
        model.config['best_val_accuaracy'] = model.config['current_val_accuracy']
        model.config['best_val_eda'] = model.config['current_val_eda']
        save_current(model)
        save_best(model)
    else:
        print('starts to overfit!')
        break
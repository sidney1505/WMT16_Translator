import code, os, shutil, tflearn, datetime, sys, git, time, json, random, copy, gensim
import numpy as np
from PIL import Image
from utils import *
from gensim.models.wrappers import FastText
#code.interact(local=dict(globals(), **locals()))

class Dataloader:
    def __init__(self, config):
        print('Create Dataloader!')
        self.config = config
        #
        data = os.listdir(self.config['dataset_path'] + '/superbatches')
        #self.coco_api = COCO(self.config['dataset_path'] + '/annotations/instances_train2017.json')
        #
        self.fast_text_en = FastText.load_fasttext_format(self.config['vocab_path'] + '/word_models/en.bin')
        reader = open(self.config['vocab_path'],'r')
        self.vocabs_en = reader.read().split('\n')
        reader.close()
        #
        self.superbatches = {}
        self.superbatches['train'] = list(filter(lambda x: x[:len('train')] == 'train', data))
        self.superbatches['val'] = list(filter(lambda x: x[:len('val')] == 'val', data))
        self.superbatches['test'] = list(filter(lambda x: x[:len('test')] == 'test', data))
        #
        self.superbatches['train'].sort()
        self.superbatches['val'].sort()
        self.superbatches['test'].sort()
        #
        self.num_superbatches = {}
        self.num_superbatches['train'] = len(self.superbatches['train'])
        self.num_superbatches['val'] = len(self.superbatches['val'])
        self.num_superbatches['test'] = len(self.superbatches['test'])
        #
        self.current_superbatch_nr = {}
        self.current_superbatch_nr['train'] = 0
        self.current_superbatch_nr['val'] = 0
        self.current_superbatch_nr['test'] = 0
        #
        self.num_batches = {}
        self.batches = {}
        self.batches['train'], self.num_batches['train'] = self.create_batches(self.superbatches['train'][0], 'train')
        self.batches['val'], self.num_batches['val'] = self.create_batches(self.superbatches['val'][0], 'val')
        self.batches['test'], self.num_batches['test'] = self.create_batches(self.superbatches['test'][0], 'test')
        #
        self.current_batch_nr = {}
        self.current_batch_nr['train'] = 0
        self.current_batch_nr['val'] = 0
        self.current_batch_nr['test'] = 0

    def create_batches(self, superbatch_name, phase='train'):
        print('Load batch ' + superbatch_name)
        #
        reader = open(self.config['dataset_path'] + '/superbatches/' + superbatch_name,'r')
        data = json.loads(reader.read())
        reader.close()
        # load the word vectors for the other language
        # TODO in future encode information of data type here too
        language = data[0]['language'] # TODO determine language first
        self.fast_text_foreign = FastText.load_fasttext_format(self.config['vocab_path'] + '/word_models/' + language + '.bin')
        # sgd + comparability
        random.seed(self.config['current_epoch'])
        random.shuffle(data)
        # split the data into batches
        def chunks(l, n):
            n = max(1, n)
            return (l[i:i+n] for i in range(0, len(l), n))
        data = chunks(data, self.config['batch_size'])
        #
        meta_batches = []
        for it, batch_sample in enumerate(data):
            if it % 10 == 0:
                print(it)
            def create_subbatches(meta_batch_data):
                batches = []
                src_vecs = []
                tgt_vecs = []
                tgt_ids = []
                names = []
                for idx, sample in enumerate(meta_batch_data):
                    names.append('no_name_defined')
                    # TODO encode the positional encodings here too!
                    src_vec = np.zeros([len(current_batch), self.config['max_sentence_len'], self.config['embedding_size']], dtype=np.float32)
                    for batch_idx, sample in enumerate(sample['src']):
                        for token_idx, word in enumerate(sample['src'].split(' ')):
                            if token_idx >= self.config['max_sentence_len']:
                                break
                            src_vec[batch_idx][token_idx] = self.fast_text_foreign.wv[word]
                    src_vecs.append(src_vec)
                    # TODO encode the positional encodings here!
                    tgt_vec = np.zeros([len(current_batch), self.config['max_sentence_len'], self.config['embedding_size']], dtype=np.float32)
                    tgt_id = np.zeros([len(current_batch), self.config['max_sentence_len']], dtype=np.int32)
                    for batch_idx, sample in enumerate(sample['tgt']):
                        for token_idx, word in enumerate(sample['tgt'].split(' ')):
                            if token_idx >= self.config['max_sentence_len']:
                                break
                            tg_vec[batch_idx][token_idx] = self.fast_text_en.wv[word]
                            if word in self.en_vocabs:
                                tgt_id[batch_idx][token_idx] = self.vocabs.index(word)
                            else:
                                tgt_id[batch_idx][token_idx] = self.config['vocab_size'] - 1 # UNK token
                    tgt_vecs.append(tgt_vec)
                    tgt_ids.append(tgt_id)
                    if (idx + 1) % self.config['batch_size'] == 0:
                        try:
                            batches.append({'src' : np.stack(src), 'tgt_vecs' : np.stack(tgt_vecs), 'tgt_ids' : np.stack(tgt_ids), 'names' : names})
                        except Exception:
                            print('error in create subbatches')
                            code.interact(local=dict(globals(), **locals()))
                        src = []
                        tgt_vecs = []
                        tgt_ids = []
                        names = []
                return batches
            meta_batch = {}
            meta_batch['train'] = create_subbatches(batch_sample['train'])
            meta_batch['val'] = create_subbatches(batch_sample['val'])
            meta_batches.append(meta_batch)
        #
        return meta_batches, len(meta_batches)

    def shuffle_train_data(self):
        random.seed(self.config['current_epoch'])
        random.shuffle(self.superbatches['train'])

    def next_batch(self, phase):
        is_finished = False
        try:
            current_batch = self.batches[phase][self.current_batch_nr[phase]]
        except Exception:
            print('next batch could not be loaded!')
            code.interact(local=dict(globals(), **locals()))
        if self.current_batch_nr[phase] >= self.num_batches[phase] - 1:
            if self.current_superbatch_nr[phase] >= self.num_superbatches[phase] - 1:
                self.current_superbatch_nr[phase] = 0
                is_finished = True
            else:
                self.current_superbatch_nr[phase] += 1
            self.batches[phase], self.num_batches[phase] = self.create_batches(self.superbatches[phase][self.current_superbatch_nr[phase]], phase)
            while self.batches[phase] == [] and not is_finished:
                print('whole batch is empty!!!')
                print('')
                if self.current_superbatch_nr[phase] >= self.num_superbatches[phase] - 1:
                    self.current_superbatch_nr[phase] = 0
                    is_finished = True
                else:
                    self.current_superbatch_nr[phase] += 1
                self.batches[phase], self.num_batches[phase] = self.create_batches(self.superbatches[phase][self.current_superbatch_nr[phase]], phase)
            self.current_batch_nr[phase] = 0
        else:
            self.current_batch_nr[phase] += 1
        return current_batch, is_finished
import json, shutil, os
import numpy as np
d = open('wmt_translation.json','r').read()
data = json.loads(d)
nr_samples = len(data)
en_vocab = {}
de_vocab = {}
shutil.rmtree('superbatches2', ignore_errors=True)
os.makedirs('superbatches2')
superbatch = []
superbatch_nr = 0
for idx, sample in enumerate(data):
    if idx % 1000 == 0:
        print(idx)
    for word in sample['src']:
        if word in de_vocab.keys():
            de_vocab[word] += 1
        else:
            de_vocab[word] = 1
    for word in sample['tgt']:
        if word in en_vocab.keys():
            en_vocab[word] += 1
        else:
            en_vocab[word] = 1
    sample['src_language'] = 'de'
    sample['tgt_language'] = 'en'
    superbatch.append(sample)
    if len(superbatch) > 10000:
        if float(idx) / nr_samples < 0.8:
            prefix = 'train_de_en'
        elif float(idx) / nr_samples < 0.9:
            prefix = 'val___de_en'
        else:
            prefix = 'test__de_en'
        num_str = str(superbatch_nr)
        padding = (3 - len(num_str)) * '0'
        open('superbatches2/' + prefix + padding + num_str + '.json', 'w').write(json.dumps(superbatch, indent=2))
        superbatch_nr += 1
        superbatch = []
idx = np.argsort(en_vocab.values())[:20000]
en_vocab_sorted = map(lambda x: en_vocab.keys()[idx[x]], range(20000))
open('en_vocab','w').write('\n'.join(en_vocab_sorted))
idx = np.argsort(de_vocab.values())[:20000]
de_vocab_sorted = map(lambda x: de_vocab.keys()[idx[x]], range(20000))
open('de_vocab','w').write('\n'.join(de_vocab_sorted))
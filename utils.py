import numpy as np
import tensorflow as tf
import json
import code
import distance


def write_log(path, value):
    writer = open(path,'w')
    writer.write(str(value))
    writer.close()

def write_config(config):
    try:
        config_string = json.dumps(config)
        config_string = config_string.replace(',',',\n').replace('{','{\n').replace('}','\n}')
        config_writer = open(config['config_path'],'w')
        config_writer.write(config_string)
        config_writer.close()
    except Exception as e:
        code.interact(local=dict(globals(), **locals()))

def load_config(config_path):
    if len(args.load_from.split('/')) == 1:
        config_path = base_config['base_model_dir'] + '/' + config_path
    if config_path[len('/config.json'):] != '/config.json':
        config_path += '/config.json'
    config_reader = open(config_path,'r')
    config_string = config_reader.read()
    config_reader.close()
    config_string = config_string.replace(',\n',',').replace('{\n','{').replace('\n}','}')
    config = json.loads(config_string)

# TODO torch style
def count_train_vars(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

# TODO torch style
def print_train_vars_light(model):
    print('Total Variables: ' + str(len(tf.trainable_variables())) + ' tensors containing ' + str(model.config['num_weights']) + ' weights!')
    for it, v in enumerate(tf.trainable_variables()):
        variable_size = np.prod(v.get_shape().as_list())
        print(str(it) + ' : ' + v.name + ' : ' + str(v.get_shape()) + ' : ' + str(variable_size) + ' (' + str(100 * float(variable_size) / model.config['num_weights']) + '%)')

# TODO torch style
def print_train_vars(model):
    print('Total Variables: ' + str(len(tf.trainable_variables())) + ' tensors containing ' + str(model.config['num_weights']) + ' weights!')
    for it, v in enumerate(tf.trainable_variables()):
        variable_size = np.prod(v.get_shape().as_list())
        try:
            weight = model.session.run(v)
            print(str(it) + ' : ' + v.name + ' : ' + str(v.get_shape()) + ' : ' + str(variable_size) + \
                ' (' + str(100 * float(variable_size) / model.config['num_weights']) + '%) : ' + str(np.mean(np.absolute(weight[it]))))
        except Exception:
            pass

# TODO torch style
def print_gradients(model, gradients, weights):
    print('Total Variables: ' + str(len(tf.trainable_variables())) + ' tensors containing ' + str(model.config['num_weights']) + ' weights!')
    for it, v in enumerate(tf.trainable_variables()):
        variable_size = np.prod(v.get_shape().as_list())
        try:
            print(str(it) + ' : ' + v.name + '    Shape: ' + str(v.get_shape()) + '   Size: ' + str(variable_size) + ' (' + str(100 * float(variable_size) / model.config['num_weights']) + '%)' + \
                '   Gradients: ' + str(np.mean(np.absolute(gradients[it]))) + ' (' + str(100 * model.config['current_learning_rate'] * np.mean(np.absolute(gradients[it])) / np.mean(np.absolute(weights[it]))) + '%)')
        except Exception:
            gradient = gradients[it][0][:weights[it].shape[0]]
            print(str(it) + ' : ' + v.name + '    Shape: ' + str(v.get_shape()) + '   Size: ' + str(variable_size) + ' (' + str(100 * float(variable_size) / model.config['num_weights']) + '%)' + \
                '   Gradients: ' + str(np.mean(np.absolute(gradient))) + ' (' + str(100 * model.config['current_learning_rate'] * np.mean(np.absolute(gradient)) / np.mean(np.absolute(weights[it]))) + '%)')


def calculate_batch_accuracy(model, pred, gt, mode='acc'):
    accuracies = []
    for batch in range(len(pred)):
        if mode == 'acc':
            accuracies.append(calculate_sentence_accuracy(model, pred[batch], gt[batch]))
        else:
            accuracies.append(calculate_sentence_eda(model, pred[batch], gt[batch]))
    return np.mean(accuracies)

def calculate_sentence_eda(model, pred, gt):
    edit_distance = int(distance.levenshtein(pred, gt))
    ref = max(len(pred),len(gt))
    return 1 - (float(edit_distance) / ref)

def calculate_sentence_accuracy(model, pred, gt):
    num_correct = 0
    num_false = 0
    for word in range(len(pred)):
        if pred[word] == 'EOS':
            break
            for word in range(word,len(pred)):
                if gt[word] == 'EOS':
                    break
                else:
                    num_false += 1
            break
        elif gt[word] == 'EOS':
            break
            for word in range(word,len(gt)):
                if pred[word] == 'EOS':
                    break
                else:
                    num_false += 1
            break
        else:
            if gt[word] == pred[word]:
                num_correct += 1
            else:
                num_false += 1
    if num_correct + num_false == 0:
        return 0
    return num_correct / float(num_correct + num_false)

def print_logs(model, groundtruth, src, phase):
    print('')
    print('Phase:       ' + phase)
    print('Name:        ' + str(model.config['model_name']))
    print('Epoch:       ' + str(model.config['current_epoch']))
    print('Step:        ' + str(model.config['current_step']))
    print('Loss:        ' + str(model.current_loss) + ' (' + str(np.mean(model.losses)) + ')')
    print('Accuracy:    ' + str(model.current_accuracy) + ' (' + str(np.mean(model.accuracies)) + ')')
    print('EDA:         ' + str(model.current_eda) + ' (' + str(np.mean(model.edas)) + ')')
    print('Input     :  ' + get_sentence_string_sum_ref(model, src[0])[0])
    gt_str, ref_len = get_sentence_string_sum_ref(model, groundtruth[0])
    print('GT:          ' + gt_str)
    print('TrainPred:   ' + get_sentence_string_sum_ref2(model, model.current_train_prediction[0], ref_len))
    print('InferPred:   ' + get_sentence_string_sum_ref2(model, model.current_infer_prediction[0], ref_len))
    print('')

def get_sentence_string_sum_ref(model, sentence):
    sentence = get_sentence_string(model, sentence).split(' ')
    if len(sentence) > 20:
        return ' '.join(sentence[:10]) + '   ...   ' + ' '.join(sentence[-10:]), len(sentence)
    else:
        return get_sentence_string(model, sentence), len(sentence)

def get_sentence_string_sum_ref2(model, sentence, ref_len):
    sentence = get_sentence_string(model, sentence).split(' ')
    if ref_len > 20:
        return ' '.join(sentence[:10]) + '   ...   ' + ' '.join(sentence[ref_len-10:ref_len])
    else:
        return ' '.join(get_sentence_string(model, sentence).split(' ')[:ref_len])

def get_sentence_string_sum(model, sentence):
    sentence = get_sentence_string(model, sentence).split(' ')
    if len(sentence) > 20:
        return ' '.join(sentence[:10]) + '   ...   ' + ' '.join(sentence[-10:])
    else:
        return get_sentence_string(model, sentence)

def get_sentence_string(model, sentence):
    s = ''
    for it in range(len(sentence)):
        if sentence[it] != 'EOS':
            s += sentence[it] + ' '
        else:
            break # EOS token
    return s[:-1]

def get_batch_tokens(model, batch, rest=1):
    sentences = []
    for it in range(batch.shape[0]):
        sentences.append(get_sentence_tokens(model, batch[it], rest))
    return sentences

def get_sentence_tokens(model, sentence, rest):
    s = []
    for it in range(len(sentence)):
        if model.config['embedding_type'] == 'word2vec':
            if np.linalg.norm(sentence[it]) > model.config['eos_threshhold']:
                val = model.dataloader.vocabs.wv.similar_by_vector(sentence[it], topn=1, restrict_vocab=model.config['vocab_size'] * rest)
                word_str = val[0][0]
                s.append(word_str)
            else:
                s.append('EOS')
        elif model.config['embedding_type'] == 'one_hot':
            if sentence[it] < model.config['vocab_size']:
                s.append(model.dataloader.vocabs[sentence[it]])
            else:
                s.append('EOS')
    return s

def restore_current(model):
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                    Restore Model!                     ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    saver = tf.train.Saver()
    saver.restore(model.session, model.config['current_weights_path'])
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                    Model restored!                    ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')

def restore_best(model):
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                  Restore best model!                  ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    saver = tf.train.Saver()
    saver.restore(model.session, model.config['best_weights_path'])
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                 Best model restored!                  ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')

def restore_seeds(model):
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                     Restore seed!                     ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    saver = tf.train.Saver()
    saver.restore(model.session, model.config['seed_weights_path'])
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                    Seed restored!                     ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')

def save_current(model):
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                      Save model!                      ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    saver = tf.train.Saver(max_to_keep=5)
    #code.interact(local=dict(globals(), **locals()))
    saver.save(model.session, model.config['current_weights_path'])
    write_config(model.config)
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                     Model saved!                      ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')

def save_best(model):
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                 Save best model!                      ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(model.session, model.config['best_weights_path'])
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                 Best Model saved!                     ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')

def save_seeds(model):
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                      Save seeds!                      ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(model.session, model.config['seed_weights_path'])
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
    print('###                     Seeds saved!                      ###')
    print('#############################################################')
    print('#############################################################')
    print('#############################################################')
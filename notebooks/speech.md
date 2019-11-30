https://github.com/kamilc/speech-recognition

https://github.com/tensorflow/models/issues/5023



```python
# %pdb
```


```python
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```


```python
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
run_config = tf.estimator.RunConfig(session_config = sess_config)
```


```python
import os
os.chdir('..')
```


```python
import numpy as np
import pandas as pd
```


```python
SAMPLING_RATE=16000
```


```python
def to_path(filename):
    return './data/' + filename
```


```python
def random_stretch(audio, params):
    """
    Stretches randomly the input audio
    """
    
    rate = random.uniform(params['random_stretch_min'], params['random_stretch_max'])
    
    return librosa.effects.time_stretch(audio, rate)
```


```python
import random

def random_shift(audio, params):
    """
    Shifts randomly the input audio to the left or the right
    """
    
    _shift = random.randrange(params['random_shift_min'], params['random_shift_max'])
    
    if _shift < 0:
        pad = (_shift * -1, 0)
    else:
        pad = (0, _shift)
    
    return np.pad(audio, pad, mode='constant')
```


```python
import glob

noise_files = glob.glob('./data/_background_noise_/*.wav')
noises = {}

def random_noise(audio, params):
    _factor = random.uniform(
        params['random_noise_factor_min'],
        params['random_noise_factor_max']
    )    
    
    if params['random_noise'] > random.uniform(0, 1):
        _path = random.choice(noise_files)
        
        if _path in noises:
            wave = noises[_path]
        else:
            if os.path.isfile(_path + '.wave.hkl'):
                wave = hkl.load(_path + '.wave.hkl').astype(np.float32)
                noises[_path] = wave
            else:
                wave, _ = librosa.load(_path, sr=SAMPLING_RATE)
                hkl.dump(wave, _path + '.wave.hkl')
                noises[_path] = wave

        noise = random_shift(
            wave,
            {
                'random_shift_min': -16000,
                'random_shift_max': 16000
            }
        )
        
        max_noise = np.max(noise[0:len(audio)])
        max_wave = np.max(audio)
        
        noise = noise * (max_wave / max_noise)
        
        return _factor * noise[0:len(audio)] + (1.0 - _factor) * audio
    else:
        return audio
```


```python
import librosa
import hickle as hkl
import os.path

def load_wave(example, absolute=False):
    row, params = example
    
    _path = row.path if absolute else to_path(row.path)
    
    if os.path.isfile(_path + '.wave.hkl'):
        wave = hkl.load(_path + '.wave.hkl').astype(np.float32)
    else:
        wave, _ = librosa.load(_path, sr=SAMPLING_RATE)
        hkl.dump(wave, _path + '.wave.hkl')

    if len(wave) <= params['max_wave_length']:
        if params['augment'] and row.path.split('/')[0] != 'voxforge':
            wave = random_noise(
                random_stretch(
                    random_shift(
                        wave,
                        params
                    ),
                    params
                ),
                params
            )
    else:
        wave = None
    
    return wave, row
```


```python
import re

from pypinyin import Style
from pypinyin import pinyin_dict
from pypinyin.style import convert as convert_style

blank = '_'

def gen_pinyin_table():
    """ 生成拼音表,长度1543
    """
    pinyin_style_tone3 = set()
    # 单字拼音库
    PINYIN_DICT = pinyin_dict.pinyin_dict
    for k, v in PINYIN_DICT.items():
        for single_pinyin in v.split(','):
            r = convert_style(single_pinyin, Style.TONE3, strict=True)
            pinyin_style_tone3.add(r)
    # 声调使用数字表示的相关拼音风格下的结果使用 5 标识轻声
    pinyin_style_tone3_list = []
    for v in pinyin_style_tone3:
        if not re.search(r'\d$', v):
            v = v + '5'
        pinyin_style_tone3_list.append(v)
    pinyin_style_tone3_list = sorted(pinyin_style_tone3_list)
    # remove 'ê1', 'ê2', 'ê3', 'ê4'
    pinyin_style_tone3_list = pinyin_style_tone3_list[:-4]
    # print(pinyin_style_tone3_list)
    # len 1543
    # print(len(pinyin_style_tone3_list))
    # 添加所有音的轻声
    pinyin_style_tone3_set = set(pinyin_style_tone3_list)
    for v in pinyin_style_tone3_list:
        pinyin_style_tone3_set.add(v[:-1] + '5')
    # len 1835
    pinyin_style_tone3_list = sorted(list(pinyin_style_tone3_set))
    # 保存
    with open('./pinyin_table.txt', 'w') as f:
        f.write('\n'.join(pinyin_style_tone3_list))
    return pinyin_style_tone3_list

pinyin_table = gen_pinyin_table()   
```


```python
from pypinyin import Style, lazy_pinyin

# 汉字
RE_HANS = re.compile(
    r'^(?:['
    r'\u3007'                  # 〇
    r'\u3400-\u4dbf'           # CJK扩展A:[3400-4DBF]
    r'\u4e00-\u9fff'           # CJK基本:[4E00-9FFF]
    r'\uf900-\ufaff'           # CJK兼容:[F900-FAFF]
    r'\U00020000-\U0002A6DF'   # CJK扩展B:[20000-2A6DF]
    r'\U0002A703-\U0002B73F'   # CJK扩展C:[2A700-2B73F]
    r'\U0002B740-\U0002B81D'   # CJK扩展D:[2B740-2B81D]
    r'\U0002F80A-\U0002FA1F'   # CJK兼容扩展:[2F800-2FA1F]
    r'])+$'
)
    
def preprocess_text(sentence, blank=' '):
    """
    预处理文本,使用` `代替所有标点符号,中文转拼音.
    """
    sentence_r = [] 
    for c in sentence:
        if RE_HANS.match(c):
            sentence_r.append(c)
        else:
            sentence_r.append(blank)
    # TODO, use '_'
    return ''.join(sentence_r)


def han_to_pinyin(sentence):
    # TODO deal with ' ' and '_'
    pinyin_style_tone3 = lazy_pinyin(sentence, style=Style.TONE3)
    pinyin_style_tone3 = [i for i in pinyin_style_tone3  if i.strip()]
    # 声调使用数字表示的相关拼音风格下的结果使用 5 标识轻声
    pinyin_style_tone3_list = []
    for v in pinyin_style_tone3:
        if not re.search(r'\d$', v):
            v = v + '5'
        pinyin_style_tone3_list.append(v)
    return ' '.join(pinyin_style_tone3_list)
```


```python
train_eval_data = pd.read_csv('./data/cv_corpus_v1/train.tsv', sep='\t')
```


```python
# 过滤非中文,所以目前的模型不支持英文识别, 也无法区别停顿
train_eval_data['sentence_cn'] = train_eval_data['sentence']
train_eval_data['sentence'] = train_eval_data['sentence'].apply(preprocess_text)
train_eval_data['sentence'] = train_eval_data['sentence'].apply(han_to_pinyin)
```


```python
#train_eval_data = train_eval_data[train_eval_data.length <= 80000]
```


```python
if not os.path.isfile('train.csv'):
    eval_data = train_eval_data.sample(n=int(len(train_eval_data) * 0.1 ))
    train_data = train_eval_data[~train_eval_data.isin(eval_data)]
    eval_data = eval_data[eval_data.path.notnull()]
    train_data = train_data[train_data.path.notnull()]
    
    train_data.to_csv('train.csv', sep='\t')
    eval_data.to_csv('eval.csv', sep='\t')
else:
    train_data = pd.read_csv('train.csv', sep='\t')
    eval_data = pd.read_csv('eval.csv', sep='\t')
```


```python
test_data = pd.read_csv('./data/cv_corpus_v1/test.tsv', sep='\t')
```


```python
# 过滤非中文,所以目前的模型不支持英文识别, 也无法区别停顿
test_data['sentence'] = test_data['sentence'].apply(preprocess_text)
test_data['sentence'] = test_data['sentence'].apply(han_to_pinyin)
```


```python
train_data['path'] = train_data['path'].apply(lambda f: 'cv_corpus_v1/clips/{}'.format(f))
```


```python
eval_data['path'] = eval_data['path'].apply(lambda f: 'cv_corpus_v1/clips/{}'.format(f))
test_data['path'] = test_data['path'].apply(lambda f: 'cv_corpus_v1/clips/{}'.format(f))
eval_data = eval_data[:10]
```


```python
if not os.path.isfile('full_train.csv'):
    train_data = train_data[['path', 'sentence']]
    train_data.to_csv('full_train.csv')
else:
    train_data = pd.read_csv('full_train.csv')
```


```python
def compute_lengths(original_lengths, params):
    """
    Computes the length of data for CTC
    """
    
    return tf.cast(
        tf.floor(
            (tf.cast(original_lengths, dtype=tf.float32) - params['n_fft']) /
                params['frame_step']
        ) + 1,
        tf.int32
    )
```


```python
def encode_labels(labels, params):
    characters = list(params['alphabet'])
    
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            characters,
            list(range(len(characters)))
        ),
        -1,
        name='char2id'
    )
    
    return table.lookup(
        tf.string_split(labels, delimiter=' ')
    )
```


```python
def decode_codes(codes, params):
    characters = list(params['alphabet'])
    
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            list(range(len(characters))),
            characters
        ),
        '',
        name='id2char'
    )
    
    return table.lookup(codes)
```


```python
import unittest
import random

from hypothesis import given, settings, note, assume, reproduce_failure, given
from hypothesis import strategies as st

def generate_sentence(length=5):
    return ' '.join(random.choices(pinyin_table, k=length))


class CoderText(unittest.TestCase):
    @given(st.builds(generate_sentence))
    @settings(deadline=None)
    def test_encode_and_decode_work(self, text):

        params = { 'alphabet': pinyin_table }
        label_ph = tf.placeholder(tf.string, shape=(1), name='text')
        codes_op = encode_labels([text], params)
        decode_op = decode_codes(codes_op, params)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer(name='init_all_tables'))

            codes,decoded = session.run(
                [codes_op,decode_op],
                {
                    label_ph: [text]
                }
            )
            self.assertEqual(text, ' '.join(map(lambda s: s.decode('UTF-8'), decoded.values)))
            self.assertEqual(codes.values.dtype, np.int32)
            self.assertEqual(len(codes.values), len(text.split()))

unittest.main(argv=['first-arg-is-ignored'], exit=False)
```

    /usr/local/lib/python3.6/dist-packages/tensorflow/contrib/lite/python/__init__.py:26: PendingDeprecationWarning: WARNING: TF Lite has moved from tf.contrib.lite to tf.lite. Please update your imports. This will be a breaking error in TensorFlow version 2.0.
      _warnings.warn(WARNING, PendingDeprecationWarning)


    
    WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
    If you depend on functionality not listed there, please file an issue.
    


    .
    ----------------------------------------------------------------------
    Ran 1 test in 1.056s
    
    OK





    <unittest.main.TestProgram at 0x7f3bde1e2e80>




```python
def decode_logits(logits, lengths, params):
    if len(tf.shape(lengths).shape) == 1:
        lengths = tf.reshape(lengths, [1])
    else:
        lengths = tf.squeeze(lengths)
        
    predicted_codes, _ = tf.nn.ctc_beam_search_decoder(
        tf.transpose(logits, (1, 0, 2)),
        lengths,
        merge_repeated=True
    )
    
    codes = tf.cast(predicted_codes[0], tf.int32)
    
    text = decode_codes(codes, params)
    
    return text, codes
```


```python
class LogMelSpectrogram(tf.layers.Layer):
    def __init__(self,
                 sampling_rate,
                 n_fft,
                 frame_step,
                 lower_edge_hertz,
                 upper_edge_hertz,
                 num_mel_bins,
                 **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.frame_step = frame_step
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins = num_mel_bins
        
    def call(self, inputs, training=True):
        stfts = tf.contrib.signal.stft(
            inputs,
            frame_length=self.n_fft,
            frame_step=self.frame_step,
            fft_length=self.n_fft,
            pad_end=False
        )
        
        power_spectrograms = tf.math.real(stfts * tf.math.conj(stfts))
        
        num_spectrogram_bins = power_spectrograms.shape[-1].value
    
        linear_to_mel_weight_matrix = tf.constant(
            np.transpose(
                librosa.filters.mel(
                    sr=self.sampling_rate,
                    n_fft=self.n_fft + 1,
                    n_mels=self.num_mel_bins,
                    fmin=self.lower_edge_hertz,
                    fmax=self.upper_edge_hertz
                )
            ),
            dtype=tf.float32
        )
        
        mel_spectrograms = tf.tensordot(
            power_spectrograms,
            linear_to_mel_weight_matrix,
            1
        )
        
        mel_spectrograms.set_shape(
            power_spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )
        
        return tf.math.log(mel_spectrograms + 1e-6)
```


```python
class AtrousConv1D(tf.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias=True,
                 kernel_initializer=tf.glorot_normal_initializer(),
                 causal=True
                ):
        super(AtrousConv1D, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.causal = causal
        
        self.conv1d = tf.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='valid' if causal else 'same',
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )
        
    def call(self, inputs):
        if self.causal:
            padding = (self.kernel_size - 1) * self.dilation_rate
            inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        
        return self.conv1d(inputs)
```


```python
class ResidualBlock(tf.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, causal, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.dilated_conv1 = AtrousConv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            causal=causal
        )
        
        self.dilated_conv2 = AtrousConv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            causal=causal
        )
        
        self.out = tf.layers.Conv1D(
            filters=filters,
            kernel_size=1
        )
        
    def call(self, inputs, training=True):
        data = tf.layers.batch_normalization(
            inputs,
            training=training
        )
        
        filters = self.dilated_conv1(data)
        gates = self.dilated_conv2(data)
        
        filters = tf.nn.tanh(filters)
        gates = tf.nn.sigmoid(gates)
        
        out = tf.nn.tanh(
            self.out(
                filters * gates
            )
        )
        
        return out + inputs, out
```


```python
class ResidualStack(tf.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rates, causal, **kwargs):
        super(ResidualStack, self).__init__(**kwargs)
        
        self.blocks = [
            ResidualBlock(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                causal=causal
            )
            for dilation_rate in dilation_rates
        ]
        
    def call(self, inputs, training=True):
        data = inputs
        skip = 0
        
        for block in self.blocks:
            data, current_skip = block(data, training=training)
            skip += current_skip

        return skip
```


```python
class SpeechNet(tf.layers.Layer):
    def __init__(self, params, **kwargs):
        super(SpeechNet, self).__init__(**kwargs)
        
        self.to_log_mel = LogMelSpectrogram(
            sampling_rate=params['sampling_rate'],
            n_fft=params['n_fft'],
            frame_step=params['frame_step'],
            lower_edge_hertz=params['lower_edge_hertz'],
            upper_edge_hertz=params['upper_edge_hertz'],
            num_mel_bins=params['num_mel_bins']
        )
        
        self.expand = tf.layers.Conv1D(
            filters=params['stack_filters'],
            kernel_size=1,
            padding='same'
        )
        
        self.stacks = [
            ResidualStack(
                filters=params['stack_filters'],
                kernel_size=params['stack_kernel_size'],
                dilation_rates=params['stack_dilation_rates'],
                causal=params['causal_convolutions']
            )
            for _ in range(params['stacks'])
        ]
        
        self.out = tf.layers.Conv1D(
            filters=len(params['alphabet']) + 1,
            kernel_size=1,
            padding='same'
        )
        
    def call(self, inputs, training=True):
        data = self.to_log_mel(inputs)
        
        data = tf.layers.batch_normalization(
            data,
            training=training
        )
        
        if len(data.shape) == 2:
            data = tf.expand_dims(data, 0)
        
        data = self.expand(data)
        
        for stack in self.stacks:
            data = stack(data, training=training)
        
        data = tf.layers.batch_normalization(
            data,
            training=training
        )
        
        return self.out(data) + 1e-8
```


```python
from multiprocessing import Pool

# 避免多次调用，内存溢出
pool = Pool()

def input_fn(input_dataset, params, load_wave_fn=load_wave):
    def _input_fn():
        """
        Returns raw audio wave along with the label
        """
        
        dataset = input_dataset
        
        print(params)
        
        if 'max_text_length' in params and params['max_text_length'] is not None:
            print('Constraining dataset to the max_text_length')
            dataset = input_dataset[input_dataset.text.str.len() < params['max_text_length']]
            
        if 'min_text_length' in params and params['min_text_length'] is not None:
            print('Constraining dataset to the min_text_length')
            dataset = input_dataset[input_dataset.text.str.len() >= params['min_text_length']]
            
        if 'max_wave_length' in params and params['max_wave_length'] is not None:
            print('Constraining dataset to the max_wave_length')
            
        print('Resulting dataset length: {}'.format(len(dataset)))
        
        def generator_fn():
            buffer = []
            
            for epoch in range(params['epochs']):
                
                if params['shuffle']:
                    _dataset = dataset.sample(frac=1)
                else:
                    _dataset = input_dataset
                    
                for _, row in _dataset.iterrows():
                    buffer.append((row, params))

                    if len(buffer) >= params['batch_size']:

                        if params['parallelize']:
                            audios = pool.map(
                                load_wave_fn,
                                buffer
                            )
                        else:
                            audios = map(
                                load_wave_fn,
                                buffer
                            )

                        for audio, row in audios:
                            if audio is not None:
                                if np.isnan(audio).any():
                                    print('SKIPPING! NaN coming from the pipeline!')
                                else:
                                    #print(row.text)
                                    yield (audio, len(audio)), row.sentence.encode()

                        buffer = []

        return tf.data.Dataset.from_generator(
                generator_fn,
                output_types=((tf.float32, tf.int32), (tf.string)),
                output_shapes=((None,()), (()))
            ) \
            .padded_batch(
                batch_size=params['batch_size'],
                padded_shapes=(
                    (tf.TensorShape([None]), tf.TensorShape(())),
                    tf.TensorShape(())
                )
            )
    
    return _input_fn
```


```python
def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        audio = features['audio']
        original_lengths = features['length']
    else:
        audio, original_lengths = features

    lengths = compute_lengths(original_lengths, params)
    
    if labels is not None:
        codes = encode_labels(labels, params)

    network = SpeechNet(params)

    is_training = mode==tf.estimator.ModeKeys.TRAIN
    
    print('Is training? {}'.format(is_training))

    logits = network(audio, training=is_training)
    text, predicted_codes = decode_logits(logits, lengths, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': logits,
            'text': tf.sparse_tensor_to_dense(
                text,
                ''
            )
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs
        )
    else:        
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=codes,
                inputs=logits,
                sequence_length=lengths,
                time_major=False,
                ignore_longer_outputs_than_inputs=True
            )
        )

        mean_edit_distance = tf.reduce_mean(
            tf.edit_distance(
                tf.cast(predicted_codes, tf.int32),
                codes
            )
        )

        distance_metric = tf.metrics.mean(mean_edit_distance)

        if mode == tf.estimator.ModeKeys.EVAL:            
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={ 'edit_distance': distance_metric }
            )

        elif mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()

            tf.summary.text(
                'train_predicted_text',
                tf.sparse_tensor_to_dense(text, '')
            )
            tf.summary.scalar('train_edit_distance', mean_edit_distance)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=global_step,
                    learning_rate=params['lr'],
                    optimizer=(params['optimizer']),
                    update_ops=update_ops,
                    clip_gradients=params['clip_gradients'],
                    summaries=[
                        "learning_rate",
                        "loss",
                        "global_gradient_norm",
                    ]
                )

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op
            )
```


```python
def experiment_name(params, excluded_keys=['alphabet', 'data', 'lr', 'clip_gradients']):

    def represent(key, value):
        if key in excluded_keys:
            return None
        else:
            if isinstance(value, list):
                return '{}_{}'.format(key, '_'.join([str(v) for v in value]))
            else:
                return '{}_{}'.format(key, value)

    parts = filter(
        lambda p: p is not None,
        [
            represent(k, params[k])
            for k in sorted(params.keys())
        ]
    )

    return '/'.join(parts)
```


```python
def dataset_params(batch_size=32,
                   epochs=50000,
                   parallelize=True,
                   max_text_length=None,
                   min_text_length=None,
                   max_wave_length=80000,
                   shuffle=True,
                   random_shift_min=-4000,
                   random_shift_max= 4000,
                   random_stretch_min=0.7,
                   random_stretch_max= 1.3,
                   random_noise=0.75,
                   random_noise_factor_min=0.2,
                   random_noise_factor_max=0.5,
                   augment=False):
    return {
        'parallelize': parallelize,
        'shuffle': shuffle,
        'max_text_length': max_text_length,
        'min_text_length': min_text_length,
        'max_wave_length': max_wave_length,
        'random_shift_min': random_shift_min,
        'random_shift_max': random_shift_max,
        'random_stretch_min': random_stretch_min,
        'random_stretch_max': random_stretch_max,
        'random_noise': random_noise,
        'random_noise_factor_min': random_noise_factor_min,
        'random_noise_factor_max': random_noise_factor_max,
        'epochs': epochs,
        'batch_size': batch_size,
        'augment': augment
    }
```


```python
def experiment_params(data,
                      optimizer='Adam',
                      lr=1e-4,
                      alphabet=" 'abcdefghijklmnopqrstuvwxyz",
                      causal_convolutions=True,
                      stack_dilation_rates= [1, 3, 9, 27, 81],
                      stacks=2,
                      stack_kernel_size= 3,
                      stack_filters= 32,
                      sampling_rate=16000,
                      n_fft=160*4,
                      frame_step=160,
                      lower_edge_hertz=0,
                      upper_edge_hertz=8000,
                      num_mel_bins=160,
                      clip_gradients=None,
                      codename='regular',
                      **kwargs):
    params = {
        'optimizer': optimizer,
        'lr': lr,
        'data': data,
        'alphabet': alphabet,
        'causal_convolutions': causal_convolutions,
        'stack_dilation_rates': stack_dilation_rates,
        'stacks': stacks,
        'stack_kernel_size': stack_kernel_size,
        'stack_filters': stack_filters,
        'sampling_rate': sampling_rate,
        'n_fft': n_fft,
        'frame_step': frame_step,
        'lower_edge_hertz': lower_edge_hertz,
        'upper_edge_hertz': upper_edge_hertz,
        'num_mel_bins': num_mel_bins,
        'clip_gradients': clip_gradients,
        'codename': codename
    }
    
    #import pdb; pdb.set_trace()
    
    if kwargs is not None and 'data' in kwargs:
        params['data'] = { **params['data'], **kwargs['data'] }
        del kwargs['data']
        
    if kwargs is not None:
        params = { **params, **kwargs }
        
    return params
```


```python
import copy

def experiment(data_params=dataset_params(), **kwargs):
    params = experiment_params(
        data_params,
        **kwargs
    )
    
#     print(params)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='stats/{}'.format(experiment_name(params)),
        params=params,
        config=run_config
    )
    
    #import pdb; pdb.set_trace()
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn(
            train_data,
            params['data']
        )
    )
    
    features = {
        "audio": tf.placeholder(dtype=tf.float32, shape=[None]),
        "length": tf.placeholder(dtype=tf.int32, shape=[])
    }
    
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features
    )
    
    best_exporter = tf.estimator.BestExporter(
      name="best_exporter",
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=5
    )
    
    eval_params = copy.deepcopy(params['data'])
    eval_params['augment'] = False
    eval_params['epochs'] = 1
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn(
            eval_data,
            eval_params
        ),
        throttle_secs=60*30,
        exporters=best_exporter
    )
    
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )
```


```python
def test(data_params=dataset_params(), **kwargs):
    params = experiment_params(
        data_params,
        **kwargs
    )
    
    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='stats/{}'.format(experiment_name(params)),
        params=params,
        config=run_config
    )
    
    eval_params = copy.deepcopy(params['data'])
    eval_params['augment'] = False
    eval_params['epochs'] = 1
    eval_params['shuffle'] = False

    estimator.evaluate(
        input_fn=input_fn(
            test_data,
            eval_params
        )
    )
```


```python
def predict(filepath, **kwargs):
    params = experiment_params(
        dataset_params(
            augment=False,
            shuffle=False,
            batch_size=1,
            epochs=1,
            parallelize=False
        ),
        **kwargs
    )
    
    dataset = pd.DataFrame(columns=['path', 'sentence'])
    dataset['path'] = [filepath]
    dataset['sentence'] = ['']
    
    print(dataset)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='stats/{}'.format(experiment_name(params)),
        params=params,
        config=run_config
    )

    return list(
        estimator.predict(
            input_fn=input_fn(
                dataset,
                params['data']
            )
        )
    )
```


```python
results = predict(
    'cv_corpus_v1/clips/common_voice_zh-CN_18531538.mp3',
    codename='deep_max_20_seconds',
    alphabet=pinyin_table, 
    causal_convolutions=False,
    stack_dilation_rates=[1, 3, 9, 27],
    stacks=6,
    stack_kernel_size=7,
    stack_filters=3*128,
    n_fft=160*8,
    frame_step=160*4,
    num_mel_bins=160,
    optimizer='Momentum',
    lr=0.00001,
    clip_gradients=20.0
)
b''.join(results[0]['text'])
```

                                                    path sentence
    0  cv_corpus_v1/clips/common_voice_zh-CN_18531538...         
    INFO:tensorflow:Using config: {'_model_dir': 'stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': gpu_options {
      per_process_gpu_memory_fraction: 0.5
      allow_growth: true
    }
    allow_soft_placement: true
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f3bccfe75f8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    {'parallelize': False, 'shuffle': False, 'max_text_length': None, 'min_text_length': None, 'max_wave_length': 80000, 'random_shift_min': -4000, 'random_shift_max': 4000, 'random_stretch_min': 0.7, 'random_stretch_max': 1.3, 'random_noise': 0.75, 'random_noise_factor_min': 0.2, 'random_noise_factor_max': 0.5, 'epochs': 1, 'batch_size': 1, 'augment': False}
    Constraining dataset to the max_wave_length
    Resulting dataset length: 1
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/dataset_ops.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, use
        tf.py_function, which takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        
    INFO:tensorflow:Calling model_fn.
    Is training? False
    WARNING:tensorflow:From <ipython-input-32-8bb2a1feb940>:41: batch_normalization (from tensorflow.python.layers.normalization) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.batch_normalization instead.
    logits: Tensor("speech_net/add:0", shape=(?, ?, 1836), dtype=float32)
    lengths: Tensor("Cast_1:0", shape=(?,), dtype=int32)
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.
    INFO:tensorflow:Restoring parameters from stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt-15291
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.





    b''



### Experiments


```python
experiment(
    dataset_params(
        parallelize=True,
        batch_size=16,
        epochs=10000,
        max_wave_length=320000,
        augment=False,
        random_noise=0.75,
        random_noise_factor_min=0.1,
        random_noise_factor_max=0.15,
        random_stretch_min=0.8,
        random_stretch_max=1.2
    ),
    codename='deep_max_20_seconds',
    alphabet = pinyin_table,
    causal_convolutions=False,
    stack_dilation_rates=[1, 3, 9, 27],
    stacks=6,
    stack_kernel_size=7,
    stack_filters=3*128,
    n_fft=160*8,
    frame_step=160*4,
    num_mel_bins=160,
    optimizer='Momentum',
    lr=0.00001,
    clip_gradients=20.0
)
```

    INFO:tensorflow:Using config: {'_model_dir': 'stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': gpu_options {
      per_process_gpu_memory_fraction: 0.5
      allow_growth: true
    }
    allow_soft_placement: true
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f3af4242320>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    INFO:tensorflow:Not using Distribute Coordinator.
    INFO:tensorflow:Running training and evaluation locally (non-distributed).
    INFO:tensorflow:Start train and evaluate loop. The evaluate will happen after every checkpoint. Checkpoint frequency is determined based on RunConfig arguments: save_checkpoints_steps None or save_checkpoints_secs 600.
    {'parallelize': True, 'shuffle': True, 'max_text_length': None, 'min_text_length': None, 'max_wave_length': 320000, 'random_shift_min': -4000, 'random_shift_max': 4000, 'random_stretch_min': 0.8, 'random_stretch_max': 1.2, 'random_noise': 0.75, 'random_noise_factor_min': 0.1, 'random_noise_factor_max': 0.15, 'epochs': 10000, 'batch_size': 16, 'augment': False}
    Constraining dataset to the max_wave_length
    Resulting dataset length: 2071
    INFO:tensorflow:Calling model_fn.
    Is training? True
    logits: Tensor("speech_net/add:0", shape=(?, ?, 1836), dtype=float32)
    lengths: Tensor("Cast_1:0", shape=(?,), dtype=int32)
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/metrics_impl.py:363: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt-15291
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:1070: get_checkpoint_mtimes (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file utilities to get mtimes.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 15291 into stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt.
    INFO:tensorflow:loss = 140.1194, step = 15291
    INFO:tensorflow:global_step/sec: 0.859657
    INFO:tensorflow:loss = 122.83241, step = 15391 (116.327 sec)
    INFO:tensorflow:global_step/sec: 0.80614
    INFO:tensorflow:loss = 134.20001, step = 15491 (124.049 sec)
    INFO:tensorflow:global_step/sec: 0.999378
    INFO:tensorflow:loss = 117.942604, step = 15591 (100.062 sec)
    INFO:tensorflow:global_step/sec: 0.964646
    INFO:tensorflow:loss = 117.81976, step = 15691 (103.665 sec)
    INFO:tensorflow:Saving checkpoints for 15792 into stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to delete files with this prefix.
    {'parallelize': True, 'shuffle': True, 'max_text_length': None, 'min_text_length': None, 'max_wave_length': 320000, 'random_shift_min': -4000, 'random_shift_max': 4000, 'random_stretch_min': 0.8, 'random_stretch_max': 1.2, 'random_noise': 0.75, 'random_noise_factor_min': 0.1, 'random_noise_factor_max': 0.15, 'epochs': 1, 'batch_size': 16, 'augment': False}
    Constraining dataset to the max_wave_length
    Resulting dataset length: 10
    INFO:tensorflow:Calling model_fn.
    Is training? False
    logits: Tensor("speech_net/add:0", shape=(?, ?, 1836), dtype=float32)
    lengths: Tensor("Cast_1:0", shape=(?,), dtype=int32)
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2019-11-25T14:57:37Z
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt-15792
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2019-11-25-14:57:40
    INFO:tensorflow:Saving dict for global step 15792: edit_distance = 0.0, global_step = 15792, loss = 0.0
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 15792: stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt-15792
    INFO:tensorflow:Loading best metric from event files.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/summary/summary_iterator.py:68: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`
    INFO:tensorflow:global_step/sec: 0.807006
    INFO:tensorflow:loss = 123.38585, step = 15791 (123.914 sec)
    INFO:tensorflow:global_step/sec: 1.02712
    INFO:tensorflow:loss = 97.64815, step = 15891 (97.360 sec)
    INFO:tensorflow:global_step/sec: 0.907419
    INFO:tensorflow:loss = 124.67094, step = 15991 (110.204 sec)
    INFO:tensorflow:global_step/sec: 0.971523
    INFO:tensorflow:loss = 110.284996, step = 16091 (102.931 sec)
    INFO:tensorflow:global_step/sec: 0.872811
    INFO:tensorflow:loss = 116.599594, step = 16191 (114.572 sec)
    INFO:tensorflow:global_step/sec: 0.931521
    INFO:tensorflow:loss = 118.370834, step = 16291 (107.351 sec)
    INFO:tensorflow:Saving checkpoints for 16392 into stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt.
    INFO:tensorflow:Skip the current checkpoint eval due to throttle secs (1800 secs).
    INFO:tensorflow:global_step/sec: 0.862026
    INFO:tensorflow:loss = 123.44051, step = 16391 (116.004 sec)
    INFO:tensorflow:global_step/sec: 0.942657
    INFO:tensorflow:loss = 109.44376, step = 16491 (106.084 sec)
    INFO:tensorflow:global_step/sec: 0.806579
    INFO:tensorflow:loss = 146.56006, step = 16591 (123.981 sec)
    INFO:tensorflow:global_step/sec: 0.766367
    INFO:tensorflow:loss = 158.31076, step = 16691 (130.485 sec)
    INFO:tensorflow:global_step/sec: 0.908245
    INFO:tensorflow:loss = 117.88295, step = 16791 (110.102 sec)
    INFO:tensorflow:global_step/sec: 0.916414
    INFO:tensorflow:loss = 133.88837, step = 16891 (109.122 sec)
    INFO:tensorflow:Saving checkpoints for 16986 into stats/causal_convolutions_False/codename_deep_max_20_seconds/frame_step_640/lower_edge_hertz_0/n_fft_1280/num_mel_bins_160/optimizer_Momentum/sampling_rate_16000/stack_dilation_rates_1_3_9_27/stack_filters_384/stack_kernel_size_7/stacks_6/upper_edge_hertz_8000/model.ckpt.
    INFO:tensorflow:Skip the current checkpoint eval due to throttle secs (1800 secs).
    INFO:tensorflow:global_step/sec: 0.928797
    INFO:tensorflow:loss = 118.173836, step = 16991 (107.666 sec)
    INFO:tensorflow:global_step/sec: 0.852818
    INFO:tensorflow:loss = 137.09431, step = 17091 (117.257 sec)



```python

```

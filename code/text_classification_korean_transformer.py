# -*- coding: utf-8 -*-
"""text_classification_korean_Transformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AjNQfkAECbwo6xmCfrKDxi8wS_hikGzq

# Text classification with Transformer

## 1. 환경설정

### 1-1. 라이브러리 및 폰트
"""

# !pip install sentencepiece
# !pip install konlpy


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from google.colab import drive
import sentencepiece as spm
import pandas as pd
import numpy as np
import os
import glob
import gzip
import shutil
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

"""### 1-2. 버전 확인"""

print(tf.__version__)
print(pd.__version__)
print(np.__version__)

"""### 1-3. GPU 확인"""

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



"""## 2. 데이터 전처리
### 2-1. 데이터 로드
"""

# drive.mount('/content/drive')
# data를 저장할 폴더 입니다. 환경에 맞게 수정 하세요.
os.chdir('../data')
data_dir = "../data/train/"
test_dir = "./test/"

"""### 2-2. 데이터 EDA"""

"""### 2-3. 데이터 전처리
- 불용어 제거 및 문장 인코딩

#### 2-2-1. SentencePiece
"""

def sentencepiece_preprocessing(train, test, vocab_size, maxlen_subject, maxlen_prompt, maxlen_paragraph):

    vocab_file = "../vocab/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    '''
    # 데이터 인코딩 테스트
    lines = [
        "겨울이 되어서 날씨가 무척 추워요.",
        "이번 성탄절은 화이트 크리스마스가 될까요?",
        "겨울에 감기 조심하시고 행복한 연말 되세요."
            ]
    for line in lines:
    pieces = vocab.encode_as_pieces(line)
    ids = vocab.encode_as_ids(line)
    print(line)
    print(pieces)
    print(ids)
    print()
    '''

    y_col = train.columns.drop(["essay_main_subject", "essay_prompt", "paragraph", 'essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp'])
    
    x_train = train[["essay_main_subject", "essay_prompt", "paragraph"]]
    y_train = train[y_col]
    x_test = test[["essay_main_subject", "essay_prompt", "paragraph"]]
    y_test = test[y_col]
    
    for l in tqdm(range(len(x_train))):
        x_train['essay_main_subject'][l] = vocab.encode_as_ids(x_train['essay_main_subject'][l])
        x_train['essay_prompt'][l] = vocab.encode_as_ids(x_train['essay_prompt'][l])
        x_train['paragraph'][l] = vocab.encode_as_ids(x_train['paragraph'][l])

    for l in tqdm(range(len(x_test))):
        x_test['essay_main_subject'][l] = vocab.encode_as_ids(x_test['essay_main_subject'][l])
        x_test['essay_prompt'][l] = vocab.encode_as_ids(x_test['essay_prompt'][l])
        x_test['paragraph'][l] = vocab.encode_as_ids(x_test['paragraph'][l])
    
    # y_train = np.asarray(y_train).astype('float32')
    # y_test = np.asarray(y_test).astype('float32')

    # 패딩
    x_train1 = keras.preprocessing.sequence.pad_sequences(x_train['essay_main_subject'], maxlen=maxlen_subject)
    x_train2 = keras.preprocessing.sequence.pad_sequences(x_train['essay_prompt'], maxlen=maxlen_prompt)
    x_train3 = keras.preprocessing.sequence.pad_sequences(x_train['paragraph'], maxlen=maxlen_paragraph)
    x_test1 = keras.preprocessing.sequence.pad_sequences(x_test['essay_main_subject'], maxlen=maxlen_subject)
    x_test2 = keras.preprocessing.sequence.pad_sequences(x_test['essay_prompt'], maxlen=maxlen_prompt)
    x_test3 = keras.preprocessing.sequence.pad_sequences(x_test['paragraph'], maxlen=maxlen_paragraph)

    return [x_train1, x_train2, x_train3], y_train, [x_test1, x_test2, x_test3], y_test, 0

    # return x_train, y_train, x_test, y_test, 0

"""#### 2-2-2. KoNLPy
- Okt, Hannanum, Kkma, Komoran 사용 가능
"""

from konlpy.tag import Okt

def konlpy_preprocessing(train, test, vocab_size, maxlen_subject, maxlen_prompt, maxlen_paragraph):

    okt = Okt()
    # stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    y_col = train.columns.drop(["essay_main_subject", "essay_prompt", "paragraph", 'essay_scoreT_org', 'essay_scoreT_cont', 'essay_scoreT_exp'])
    
    x_train = train[["essay_main_subject", "essay_prompt", "paragraph"]]
    y_train = train[y_col]
    x_test = test[["essay_main_subject", "essay_prompt", "paragraph"]]
    y_test = test[y_col]
    
    # for sentence in tqdm(train['document']):
    #    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    #    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    #    X_train.append(stopwords_removed_sentence)

    
    for l in tqdm(range(len(x_train))):
        x_train['essay_main_subject'][l] = okt.morphs(x_train['essay_main_subject'][l], stem=True)
        x_train['essay_prompt'][l] = okt.morphs(x_train['essay_prompt'][l], stem=True)
        x_train['paragraph'][l] = okt.morphs(x_train['paragraph'][l], stem=True)

    for l in tqdm(range(len(x_test))):
        x_test['essay_main_subject'][l] = okt.morphs(x_test['essay_main_subject'][l], stem=True)
        x_test['essay_prompt'][l] = okt.morphs(x_test['essay_prompt'][l], stem=True)
        x_test['paragraph'][l] = okt.morphs(x_test['paragraph'][l], stem=True)

    tokenizer1 = Tokenizer() # 정수 인코딩
    tokenizer1.fit_on_texts(x_train['essay_main_subject']) # tokenizer.word_index로 확인 가능
    tokenizer2 = Tokenizer() # 정수 인코딩
    tokenizer2.fit_on_texts(x_train['essay_prompt']) # tokenizer.word_index로 확인 가능
    tokenizer3 = Tokenizer() # 정수 인코딩
    tokenizer3.fit_on_texts(x_train['paragraph']) # tokenizer.word_index로 확인 가능

    x_train['essay_main_subject'] = tokenizer1.texts_to_sequences(x_train['essay_main_subject'])
    x_train['essay_prompt'] = tokenizer2.texts_to_sequences(x_train['essay_prompt'])
    x_train['paragraph'] = tokenizer3.texts_to_sequences(x_train['paragraph'])
    x_test['essay_main_subject'] = tokenizer1.texts_to_sequences(x_test['essay_main_subject'])
    x_test['essay_prompt'] = tokenizer2.texts_to_sequences(x_test['essay_prompt'])
    x_test['paragraph'] = tokenizer3.texts_to_sequences(x_test['paragraph'])

    # 패딩
    x_train1 = keras.preprocessing.sequence.pad_sequences(x_train['essay_main_subject'], maxlen=maxlen_subject)
    x_train2 = keras.preprocessing.sequence.pad_sequences(x_train['essay_prompt'], maxlen=maxlen_prompt)
    x_train3 = keras.preprocessing.sequence.pad_sequences(x_train['paragraph'], maxlen=maxlen_paragraph)
    x_test1 = keras.preprocessing.sequence.pad_sequences(x_test['essay_main_subject'], maxlen=maxlen_subject)
    x_test2 = keras.preprocessing.sequence.pad_sequences(x_test['essay_prompt'], maxlen=maxlen_prompt)
    x_test3 = keras.preprocessing.sequence.pad_sequences(x_test['paragraph'], maxlen=maxlen_paragraph)

    return [x_train1, x_train2, x_train3], y_train, [x_test1, x_test2, x_test3], y_test, [tokenizer1, tokenizer2, tokenizer3]

def data_preprocessing(method, train, test, maxlen):
    
    vocab_size = 20000  # Only consider the top 20k words
    # maxlen = 200  # Only consider the first 200 words of each movie review

    if method == 'SentencePiece':
        return sentencepiece_preprocessing(train, test, vocab_size, maxlen[0], maxlen[1], maxlen[2])
    
    elif method == 'konlpy':
        return konlpy_preprocessing(train, test, vocab_size, maxlen[0], maxlen[1], maxlen[2])
    
    else:
        print('None')

"""## 3. 모델

### 3-1. 모델

#### 3-1-1. LSTM

#### 3-1-2. Attention
"""

from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.layers import Layer

class Attention(keras.layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
          'W_regularizer': self.W_regularizer,
          'b_regularizer': self.b_regularizer,
          'W_constraint': self.W_constraint,
          'b_constraint': self.b_constraint,
          'bias': self.bias
        })
        return config

      
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

"""#### 3-1-3. Transformer
- Transformer 블록을 레이어로 사용
"""

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

"""- 임베딩 레이어"""

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

"""### 3-2. 모델 구현 함수"""

def make_model(model_name, maxlen, vocab_size):
    embed_dim = 32  # Embedding size for each token

    y_cols = ['essay_scoreT_org_0', 'essay_scoreT_org_1', 'essay_scoreT_org_2',
                'essay_scoreT_org_3', 'essay_scoreT_cont_0', 'essay_scoreT_cont_1',
                'essay_scoreT_cont_2', 'essay_scoreT_cont_3', 'essay_scoreT_exp_0',
                'essay_scoreT_exp_1', 'essay_scoreT_exp_2']

    outputs = []

    if model_name == 'LSTM':
        inputs = layers.Input(shape=(maxlen,))
        x = layers.Embedding(vocab_size, embed_dim)(inputs)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="softmax")(x)
    
    elif model_name == 'Attention':
        inputs_subject = layers.Input(shape=(maxlen[0],))
        inputs_prompt = layers.Input(shape=(maxlen[1],))
        inputs_paragraph = layers.Input(shape=(maxlen[2],))

        x1 = layers.Embedding(vocab_size, embed_dim)(inputs_subject)
        x1 = Attention(maxlen[0])(x1)
        
        x2 = layers.Embedding(vocab_size, embed_dim)(inputs_prompt)
        x2 = Attention(maxlen[1])(x2)

        x3 = layers.Embedding(vocab_size, embed_dim)(inputs_paragraph)
        x3 = Attention(maxlen[2])(x3)

        x = layers.concatenate([x1, x2, x3])
        # x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.1)(x)

        for y_col in y_cols:
            outputs.append(layers.Dense(4, activation="softmax", name=y_col)(x))

    elif model_name == 'Transformer':
        num_heads = 4  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs_subject = layers.Input(shape=(maxlen[0],))
        inputs_prompt = layers.Input(shape=(maxlen[1],))
        inputs_paragraph = layers.Input(shape=(maxlen[2],))

        embedding_layer1 = TokenAndPositionEmbedding(maxlen[0], vocab_size, embed_dim)
        x1 = embedding_layer1(inputs_subject)
        transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)

        x1 = transformer_block1(x1)
        x1 = layers.GlobalAveragePooling1D()(x1)
        x1 = layers.Dropout(0.1)(x1)

        embedding_layer2 = TokenAndPositionEmbedding(maxlen[1], vocab_size, embed_dim)
        x2 = embedding_layer2(inputs_prompt)
        transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)

        x2 = transformer_block2(x2)
        x2 = layers.GlobalAveragePooling1D()(x2)
        x2 = layers.Dropout(0.1)(x2)

        embedding_layer3 = TokenAndPositionEmbedding(maxlen[2], vocab_size, embed_dim)
        x3 = embedding_layer3(inputs_paragraph)
        transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim)

        x3 = transformer_block3(x3)
        x3 = layers.GlobalAveragePooling1D()(x3)
        x3 = layers.Dropout(0.1)(x3)

        c = layers.concatenate([x1, x2, x3])
        c = layers.Dense(64, activation="relu")(c)
        c = layers.Dropout(0.1)(c)

        for y_col in y_cols:
            outputs.append(layers.Dense(4, activation="softmax", name=y_col)(c))

    else:
        inputs = layers.Input(shape=(maxlen,))
        inputs_test = layers.Input(shape=(10,))

        x = layers.Embedding(vocab_size, embed_dim)(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)

        x_test = layers.Embedding(vocab_size, embed_dim)(inputs_test)
        x_test = layers.GlobalAveragePooling1D()(x_test)
        x_test = layers.Dropout(0.1)(x_test)

        c = layers.concatenate([x, x_test])
        c = layers.Dense(64, activation="relu")(c)
        c = layers.Dropout(0.1)(c)
        c = layers.Dense(32, activation="relu")(c)
        c = layers.Dropout(0.1)(c)

        outputs = layers.Dense(2, activation="softmax")(c)


    return keras.Model(inputs=[inputs_subject, inputs_prompt, inputs_paragraph], outputs=outputs)

"""### 3-3. 모델 실행 함수"""

def model_run(model, x_train, y_train, checkpoint_path, optimizer="adam"):
  epochs = 500
  batch_size = 16

  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001, mode='min'
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, verbose=1),
  ]
  model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
  )
  history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
  )

"""### 3-4. 모델 학습"""

def preprocessing_and_modeling(essay_type, model_name, preprocessing_method, model_algorithm, maxlen, vocab_size):
    data_dir = r"D:\\Git Repository\\AI-Writing-Supporter\\data\\train\\"
    test_dir = r"D:\\Git Repository\\AI-Writing-Supporter\\data\\test\\"
    if essay_type == '주장':
        train = pd.read_csv(data_dir+'train_주장.csv')
        test = pd.read_csv(test_dir+'test_주장.csv')
    elif essay_type == '설명글':
        train = pd.read_csv(data_dir+'train_설명글.csv')
        test = pd.read_csv(test_dir+'test_설명글.csv')
    elif essay_type == '찬성반대':
        train = pd.read_csv(data_dir+'train_찬성반대.csv')
        test = pd.read_csv(test_dir+'test_찬성반대.csv')
    elif essay_type == '대안제시':
        train = pd.read_csv(data_dir+'train_대안제시.csv')
        test = pd.read_csv(test_dir+'test_대안제시.csv')
    elif essay_type == '글짓기':
        train = pd.read_csv(data_dir+'train_글짓기.csv')
        test = pd.read_csv(test_dir+'test_글짓기.csv')

    print(train.shape, test.shape)
    print(train.head())

    train_pp = train.reset_index().drop('index', axis=1)
    test_pp = test.reset_index().drop('index', axis=1)

    x_train, y_train, x_test, y_test, tokenizer_data = data_preprocessing(preprocessing_method, train_pp, test_pp, maxlen)
    # print(x_train[0].shape, x_train[1].shape, x_train[2].shape, y_train[0].shape, y_train[1].shape, y_train[2].shape, x_test.shape, y_test.shape)

    # 모델 저장 경로
    model_dir = '../model/'+model_name +'/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # 체크포인트 파일 저장 경로
    checkpoint_path = model_dir+"checkpoints_"+model_name+".ckpt"

    model = make_model(model_algorithm, maxlen, vocab_size)
    print(model.summary())

    # plot_model(model, to_file=model_name+'.png')
    # plot_model(model, to_file=model_name+'_shapes.png', show_shapes=True)

    y_train_list = [y_train['essay_scoreT_org_0'], y_train['essay_scoreT_org_1'], y_train['essay_scoreT_org_2'],
                y_train['essay_scoreT_org_3'], y_train['essay_scoreT_cont_0'], y_train['essay_scoreT_cont_1'],
                y_train['essay_scoreT_cont_2'], y_train['essay_scoreT_cont_3'], y_train['essay_scoreT_exp_0'],
                y_train['essay_scoreT_exp_1'], y_train['essay_scoreT_exp_2']]

    model_run(  model,
                x_train, 
                y_train_list, 
                checkpoint_path, 
                optimizer="adam")

essay_types = ['주장', '설명글', '찬성반대', '대안제시', '글짓기']
for e_type in essay_types:
    preprocessing_and_modeling(e_type, 'model_trans_sp_'+e_type+'_221204_1', 'SentencePiece', 'Transformer', [20, 200, 2000], 20000)



'''
"""## 4. 성능 평가

### 4-1. 성능 평가 함수
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evalution(x_test, y_test, model):

    # data_pre = model.predict(x_test)
    # print(data_pre)
    pred_roc = model.predict(x_test)
    pred_z = []
    for i in range(11):
        pred_z_tmp = []
        for j in range(len(pred_roc[i])):
            pred_z_tmp.append(list(pred_roc[i][j]).index(max(list(pred_roc[i][j]))))
        pred_z.append(pred_z_tmp)

    pred_z = np.asarray(pred_z).astype('float32')
    y_col = y_test.columns

    for i in range(11):
        print(str(i)+'번째 평가항목')
        print(accuracy_score(y_test[y_col[i]], pred_z[i]))
        # print(precision_score(y_test[y_col[i]], pred_z[i]))
        # print(recall_score(y_test[y_col[i]], pred_z[i]))
        # print(roc_auc_score(y_test[y_col[i]], pred_z[i]))

    # test_loss, test_acc = model.evaluate(x_test, y_test)

    return pred_z, y_test
    # print("Test accuracy", test_acc)
    # print("Test loss", test_loss)

"""### 4-2. 모델 로드"""

model = make_model('Attention', maxlen, vocab_size)
model.load_weights(checkpoint_path)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

"""### 4-3. 모델 성능 평가"""

pred, y_t = evalution(x_test, y_test, model)

print(accuracy_score(list(y_t),pred))
print(precision_score(list(y_t),pred))
print(recall_score(list(y_t),pred))
print(roc_auc_score(list(y_t),pred))



"""Tensor2Tensor 있는지 확인 필요
https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb#scrollTo=OJKU36QAfqOC
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_col = y_test.columns
for i in range(11):
    print(str(i)+'번째 평가항목')
    print(classification_report(y_test[y_col[i]], pred[i]))





















pred_df = pd.DataFrame({"pred":pred})

pred_df["pred"].value_counts()

y_tdf = pd.DataFrame({"pred2":y_t})

y_tdf["pred2"].value_counts()

df_cc = pd.DataFrame({"pred2":pred_df["pred"] + y_tdf["pred2"]})

len(df_cc[df_cc["pred2"]==1.0])

df_cc2 = pd.concat([pred_df, y_tdf],axis=1)

df_cc21 = df_cc2[df_cc2["pred"]==1.0]

len(df_cc21[df_cc21["pred2"]==0.0])



train["label"].value_counts()

test["label"].value_counts()





"역시 B형이라 그런지 게으르네"
"단맛의 라떼. 호불호 갈릴듯!"
"니거는 거두면 안된다."
"보고 싶은 영화입니다 좋은 영화 리뷰네요"
"어휴.. 너무 안타깝네요"

line_input_data = input()

line_input = []
# line_input.append("어휴.. 너무 억울하겠다")
line_input.append(line_input_data)

lines = pd.DataFrame({"comments":list(line_input), "label": [0]})

line_test = lines.copy()

_, _, line_x, _ = data_preprocessing(line_test, lines)

line_x = keras.preprocessing.sequence.pad_sequences(line_x, maxlen=maxlen)

pred = model.predict(line_x)
pred

'''
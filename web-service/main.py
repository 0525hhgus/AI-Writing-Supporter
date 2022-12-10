import flask
from flask import Flask, request, render_template
from tensorflow import keras
import sentencepiece as spm
import pandas as pd
import korean_essay_eval_transfromer as keet

import tensorflow as tf
from tensorflow import keras

data_dir = "../data/train/"
app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


def web_sentencepiece_preprocessing(text, maxlen_subject, maxlen_prompt, maxlen_paragraph):
    vocab_file = "./kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    for l in range(len(text)):
        text['essay_main_subject'][l] = vocab.encode_as_ids(text['essay_main_subject'][l])
        text['essay_prompt'][l] = vocab.encode_as_ids(text['essay_prompt'][l])
        text['paragraph'][l] = vocab.encode_as_ids(text['paragraph'][l])

    # 패딩
    x_text1 = keras.preprocessing.sequence.pad_sequences(text['essay_main_subject'], maxlen=maxlen_subject)
    x_text2 = keras.preprocessing.sequence.pad_sequences(text['essay_prompt'], maxlen=maxlen_prompt)
    x_text3 = keras.preprocessing.sequence.pad_sequences(text['paragraph'], maxlen=maxlen_paragraph)

    return [x_text1, x_text2, x_text3]


# 데이터 예측 처리
maxlen = [20, 200, 2000]
vocab_size = 20000

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기

        essay_type = str(request.form['essay_type'])
        essay_types = ['주장', '설명글', '찬성반대', '대안제시', '글짓기']
        model_name = 'model_transformer_'+essay_types[int(essay_type[-1])-1]
        print(model_name)
        text1= request.form['inputs_subject']
        text2= request.form['inputs_prompt']
        text3= request.form['inputs_paragraph']
        text = pd.DataFrame([str(text1)])
        text = pd.concat([text, pd.DataFrame([str(text2)])], axis=1)
        text = pd.concat([text, pd.DataFrame([str(text3)])], axis=1)
        text.columns = ['essay_main_subject', 'essay_prompt', 'paragraph']
        print(text)

        x_text = web_sentencepiece_preprocessing(text, maxlen[0], maxlen[1], maxlen[2])

        model = keet.make_model('Transformer', maxlen, vocab_size)

        checkpoint_path = ''
        if int(essay_type[-1]) == 1:
            checkpoint_path = './model/model_transformer_주장/checkpoints_model_transformer_주장.ckpt'
        elif int(essay_type[-1]) == 2:
            checkpoint_path = './model/model_transformer_설명글/checkpoints_model_transformer_설명글.ckpt'
        elif int(essay_type[-1]) == 3:
            checkpoint_path = './model/model_transformer_찬성반대/checkpoints_model_transformer_찬성반대.ckpt'
        elif int(essay_type[-1]) == 4:
            checkpoint_path = './model/model_transformer_대안제시/checkpoints_model_transformer_대안제시.ckpt'
        elif int(essay_type[-1]) == 5:
            checkpoint_path = './model/model_transformer_글짓기/checkpoints_model_transformer_글짓기.ckpt'

        # checkpoint_path = "./model/" + model_name + "/checkpoints_" + model_name + ".ckpt"
        print(checkpoint_path)
        model.load_weights(checkpoint_path)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        # 입력 받은 텍스트 예측
        score = model.predict(x_text)

        print(score)

        # 결과 리턴
        return render_template('index.html', label=score)

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    maxlen = [20, 200, 2000]
    vocab_size = 20000

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)

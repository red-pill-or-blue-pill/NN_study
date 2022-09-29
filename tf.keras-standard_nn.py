# （必要に応じて）座標点データを生成するライブラリをインストールする必要がある
#!pip install playground-data

# playground-dataライブラリのplygdataパッケージを「pg」という別名でインポート
import plygdata as pg

# 問題種別で「分類（Classification）」を選択し、
# データ種別で「2つのガウシアンデータ（TwoGaussData）」を選択する場合の、
# 設定値を定数として定義
PROBLEM_DATA_TYPE = pg.DatasetType.ClassifyTwoGaussData

# 各種設定を定数として定義
TRAINING_DATA_RATIO = 0.5  # データの何％を訓練【Training】用に？ (残りは精度検証【Validation】用) ： 50％
DATA_NOISE = 0.0           # ノイズ： 0％

# 定義済みの定数を引数に指定して、データを生成する
data_list = pg.generate_data(PROBLEM_DATA_TYPE, DATA_NOISE)

# データを「訓練用」と「精度検証用」を指定の比率で分割し、さらにそれぞれを「データ（X）」と「教師ラベル（y）」に分ける
X_train, y_train, X_valid, y_valid = pg.split_data(data_list, training_size=TRAINING_DATA_RATIO)

####################################################################################################
print('X_train:'); print(X_train[:5])
print('y_train:'); print(y_train[:5])
print('X_valid:'); print(X_valid[:5])
print('y_valid:'); print(y_valid[:5])


####################################################################################################
# ライブラリ「TensorFlow」のtensorflowパッケージを「tf」という別名でインポート
import tensorflow as tf

# 定数（モデル定義時に必要となる数値）
INPUT_FEATURES = 2      # 入力（特徴）の数： 2
LAYER1_NEURONS = 3      # ニューロンの数： 3
LAYER2_NEURONS = 3      # ニューロンの数： 3
OUTPUT_RESULTS = 1      # 出力結果の数： 1
ACTIVATION = 'tanh'     # 活性化関数（ここを書き換える）： tanh関数

# 積層型のモデルの定義
model = tf.keras.models.Sequential([

  # 隠れ層：1つ目のレイヤー
  tf.keras.layers.Dense(
    input_shape=(INPUT_FEATURES,),       # 入力の形状（＝入力層）
    units=LAYER1_NEURONS,                # ユニットの数
    activation=ACTIVATION),              # 活性化関数

  # 隠れ層：2つ目のレイヤー
  tf.keras.layers.Dense(
    units=LAYER2_NEURONS,                # ユニットの数
    activation=ACTIVATION),              # 活性化関数

  # 出力層
  tf.keras.layers.Dense(
    units=OUTPUT_RESULTS,                # ユニットの数
    activation='tanh'),                  # 活性化関数
])


####################################################################################################
import tensorflow.keras.backend as K

def tanh_accuracy(y_true, y_pred):           # y_trueは正解、y_predは予測（出力）
  threshold = K.cast(0.0, y_pred.dtype)              # -1か1かを分ける閾値を作成
  y_pred = K.cast(y_pred >= threshold, y_pred.dtype) # 閾値未満で0、以上で1に変換
  # 2倍して-1.0することで、0／1を-1.0／1.0にスケール変換して正解率を計算
  return K.mean(K.equal(y_true, y_pred * 2 - 1.0), axis=-1)



####################################################################################################
# 定数（学習方法設計時に必要となるもの）
LOSS = 'mean_squared_error'          # 損失関数： 平均二乗誤差
OPTIMIZER = tf.keras.optimizers.SGD  # 最適化： 確率的勾配降下法
LEARNING_RATE = 0.03                  # 学習率： 0.03

# モデルを生成する
model.compile(optimizer=OPTIMIZER(learning_rate=LEARNING_RATE),
              loss=LOSS,
              metrics=[tanh_accuracy])  # 精度（正解率）





####################################################################################################
# 定数（学習方法設計時に必要となるもの）
BATCH_SIZE = 15  # バッチサイズ： 15（選択肢は「1」～「30」）
EPOCHS = 100     # エポック数： 100


# 早期終了
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# CSVロガー
csv_logger = tf.keras.callbacks.CSVLogger('training.log')


# 学習する
hist = model.fit(x=X_train,                          # 訓練用データ
                 y=y_train,                          # 訓練用ラベル
                 validation_data=(X_valid, y_valid), # 精度検証用
                 batch_size=BATCH_SIZE,              # バッチサイズ
                 epochs=EPOCHS,                      # エポック数
                 verbose=1,                          # 実行状況表示
                 callbacks=[es, csv_logger])         # コールバック



####################################################################################################
import matplotlib.pyplot as plt

# 学習結果（損失）のグラフを描画
train_loss = hist.history['loss']
valid_loss = hist.history['val_loss']
epochs = len(train_loss)
plt.plot(range(epochs), train_loss, marker='.', label='loss (Training data)')
plt.plot(range(epochs), valid_loss, marker='.', label='loss (Validation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()





####################################################################################################


import plygdata as pg
import numpy as np

# 未知のテストデータを生成
PROBLEM_DATA_TYPE = pg.DatasetType.ClassifyTwoGaussData
TEST_DATA_RATIO = 1.0  # データの何％を訓練【Training】用に？ (残りは精度検証【Validation】用) ： 100％
DATA_NOISE = 0.0       # ノイズ： 0％
data_list = pg.generate_data(PROBLEM_DATA_TYPE, DATA_NOISE)
X_test, y_test, _, _ = pg.split_data(data_list, training_size=TEST_DATA_RATIO)

# 学習済みモデルを使って推論
result_proba = model.predict(X_test)
result_class = np.frompyfunc(lambda x: 1 if x >= 0.0 else -1, 1, 1)(result_proba) # 離散化
# それぞれ5件ずつ出力
print('proba:'); print(result_proba[:5])  # 予測
print('class:'); print(result_class[:5])  # 分類

# 未知のテストデータで学習済みモデルの汎化性能を評価
score = model.evaluate(X_test, y_test)
print('test loss:', score[0])  # 損失
print('test acc:', score[1])   # 正解率


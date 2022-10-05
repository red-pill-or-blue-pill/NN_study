######## DNN02.py ########
# ニューラルネットワークは3層構成
layers = [
    2,  # 入力層の入力（特徴量）の数
    3,  # 隠れ層1のノード（ニューロン）の数
    1]  # 出力層のノードの数

# 重みとバイアスの初期値
weights = [
    #[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # 入力層→隠れ層1
    [[0.2, 0.4], [0.8, 0.5], [0.2, 0.4]], # 入力層→隠れ層1
    #[[0.0, 0.0, 0.0]] # 隠れ層1→出力層
    [[0.7, 0.3, 0.4]] # 隠れ層1→出力層
]
biases = [
    [0.0, 0.0, 0.0],  # 隠れ層1
    [0.0]  # 出力層
]

# モデルを定義
model = (layers, weights, biases)

# 仮の訓練データ（1件分）を準備
x = [0.05, 0.1]  # x_1とx_2の2つの特徴量
#x = [1, 1]  # x_1とx_2の2つの特徴量

######## DNN03.py ########
import math

def summation(x,weights, bias):
    " 重み付き線形和の関数。"
	# 線形代数を使う場合のコード例：
	# linear_sum = np.dot(x, weights) + bias

	# list 4 #
    # ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    linear_sum = 0.0
    for x_i, w_i in zip(x, weights):
        linear_sum += x_i * w_i  # iは「番号」（数学は基本的に1スタート）
    linear_sum += bias
    return linear_sum

def sigmoid(x):
    " シグモイド関数。"
	# list 6 #
    return 1.0 / (1.0 + math.exp(-x))
# 線形代数の場合はmathをnpに変える（事前にimport numpy as np）

def identity(x):
    " 恒等関数。"
	# list 8 #
    return x

######## DNN10.py ########
def forward_prop(layers, weights, biases, x, cache_mode=False):
    """
    順伝播を行う関数。
    - 引数：
    (layers, weights, biases)： モデルを指定する。
    x： 入力データを指定する。
    cache_mode： 予測時はFalse、訓練時はTrueにする。これにより戻り値が変わる。
    - 戻り値：
    cache_modeがFalse時は予測値のみを返す。True時は、予測値だけでなく、
        キャッシュに記録済みの線形和（Σ）値と、活性化関数の出力値も返す。
    """

    cached_sums = []  # 記録した全ノードの線形和（Σ）の値
    cached_outs = []  # 記録した全ノードの活性化関数の出力値

    # まずは、入力層を順伝播する
    cached_outs.append(x)  # 何も処理せずに出力値を記録
    next_x = x  # 現在の層の出力（x）＝次の層への入力（next_x）

    # 次に、隠れ層や出力層を順伝播する
    SKIP_INPUT_LAYER = 1
    for layer_i, layer in enumerate(layers):  # 各層を処理
        if layer_i == 0:
            continue  # 入力層は上で処理済み

        # 各層のノードごとに処理を行う
        sums = []
        outs = []
        for node_i in range(layer):  # 層の中の各ノードを処理

            # ノードごとの重みとバイアスを取得
            w = weights[layer_i - SKIP_INPUT_LAYER][node_i]
            b = biases[layer_i - SKIP_INPUT_LAYER][node_i]

            # 【リスト3のコード】ここから↓
            # 1つのノードの処理（1）： 重み付き線形和
            node_sum = summation(next_x, w, b)

            # 1つのノードの処理（2）： 活性化関数
            if layer_i < len(layers)-1:  # -1は出力層以外の意味
                # 隠れ層（シグモイド関数）
                node_out = sigmoid(node_sum)
            else:
                # 出力層（恒等関数）
                node_out = identity(node_sum)
            # 【リスト3のコード】ここまで↑

            # 各ノードの線形和と（活性化関数の）出力をリストにまとめていく
            sums.append(node_sum)
            outs.append(node_out)

        # 各層内の全ノードの線形和と出力を記録
        cached_sums.append(sums)
        cached_outs.append(outs)
        next_x = outs  # 現在の層の出力（outs）＝次の層への入力（next_x）

    if cache_mode:
        return (cached_outs[-1], cached_outs, cached_sums)

    return cached_outs[-1]


# 訓練時の（1）順伝播の実行例
y_pred, cached_outs, cached_sums = forward_prop(*model, x, cache_mode=True)
# ※先ほど作成したモデルと訓練データを引数で受け取るよう改変した

print(f'y_pred={y_pred}')
print(f'cached_outs={cached_outs}')
print(f'cached_sums={cached_sums}')
# 出力例：
# cached_outs=[[0.05, 0.1], [0.5, 0.5, 0.5], [0.0]]  # 入力層／隠れ層1／出力層
# cached_sums=[[0.0, 0.0, 0.0], [0.0]]  # 隠れ層1／出力層（※入力層はない）


######## DNN11.py ########
# 異なるDNNアーキテクチャーを定義してみる
layers2 = [
    2,  # 入力層の入力（特徴量）の数
    3,  # 隠れ層1のノード（ニューロン）の数
    2,  # 隠れ層2のノード（ニューロン）の数
    1]  # 出力層のノードの数

# 重みとバイアスの初期値
weights2 = [
    [[-0.2, 0.4], [-0.4, -0.5], [-0.4, -0.5]], # 入力層→隠れ層1
    [[-0.2, 0.4, 0.9], [-0.4, -0.5, -0.2]], # 隠れ層1→隠れ層2
    [[-0.5, 1.0]] # 隠れ層2→出力層
]
biases2 = [
    [0.1, -0.1, 0.1],  # 隠れ層1
    [0.2, -0.2],  # 隠れ層2
    [0.3]  # 出力層
]

# モデルを定義
model2 = (layers2, weights2, biases2)

# 仮の訓練データ（1件分）を準備
x2 = [2.3, 1.5]  # x_1とx_2の2つの特徴量

# 予測時の（1）順伝播の実行例
y_pred = forward_prop(*model2, x2)
print(f'y_pred={y_pred}')  # 予測値
# 出力例：
# [0.3828840428423274]


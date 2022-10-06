import math

######## DNN03 ########
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

######## DNN5 ########
## リスト5　重み付き線形和の偏導関数の実装 ##
## 引数with_respect_toについては、
## 例えば「変数xに関しての関数f(x,w,b)の偏導関数」
## （＝関数f(x,w,b)を変数xで偏微分すること）を、
## 英語で「partial derivative of f(x,w,b) with respect to x」と
## 表現するため、このように命名した。
def sum_der(x, weights, bias, with_respect_to='w'):
    # ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    if with_respect_to == 'w':
        return x  # 線形和uを各重みw_iで偏微分するとx_iになる（iはノード番号）
    elif with_respect_to == 'b':
        return 1.0  # 線形和uをバイアスbで偏微分すると1になる
    elif with_respect_to == 'x':
        return weights  # 線形和uを各入力x_iで偏微分するとw_iになる

######## DNN6 ########
## 活性化関数（シグモイド関数）の実装 ##
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

######## DNN7 ########
## 活性化関数（シグモイド関数）の導関数の実装 ##
def sigmoid_der(x):
    output = sigmoid(x)
    return output * (1.0 - output)

######## DNN8 ########
## 活性化関数（恒等関数）の実装 ##
def identity(x):
    return x

######## DNN9 ########
## 活性化関数（恒等関数）の導関数の実装 ##
def identity_der(x):
    return 1.0

######## DNN10 ########
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

######## DNN13 ########
## 損失関数（二乗和誤差）の実装（予測値から正解値を引く式）##
def sseloss(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2

######## DNN14 ########
## 損失関数の偏導関数の実装（sseloss()関数をy_predで偏微分したもの）##
def sseloss_der(y_pred, y_true):
    return y_pred - y_true

######## DNN21 ########
## 逆伝播の処理全体の実装 ##
def back_prop(layers, weights, biases, y_true, cached_outs, cached_sums):
    """
    逆伝播を行う関数。
    - 引数：
    (layers, weights, biases)： モデルを指定する。
    y_true： 正解値（出力層のノードが複数ある場合もあるのでリスト値）。
    cached_outs： 順伝播で記録した活性化関数の出力値。予測値を含む。
    cached_sums： 順伝播で記録した線形和（Σ）値。
    - 戻り値：
    重みの勾配とバイアスの勾配を返す。
    """

    # ネットワーク全体で勾配を保持するためのリスト
    grads_w =[]  # 重みの勾配
    grads_b = []  # バイアスの勾配
    grads_x = []  # 入力の勾配

    layer_count = len(layers)
    layer_max_i = layer_count-1
    SKIP_INPUT_LAYER = 1
    PREV_LAYER = 1
    rng = range(SKIP_INPUT_LAYER, layer_count)  # 入力層以外の層インデックス
    for layer_i in reversed(rng):  # 各層を逆順に処理

        is_output_layer = (layer_i == layer_max_i)
        # 層ごとで勾配を保持するためのリスト
        layer_grads_w = []
        layer_grads_b = []
        layer_grads_x = []

        # （1）逆伝播していく誤差情報
        if is_output_layer:
            # 出力層（損失関数の偏微分係数）
            back_error = []  # 逆伝播していく誤差情報
            y_pred = cached_outs[layer_i]
            for output, target in zip(y_pred, y_true):
                loss_der = sseloss_der(output, target)  # 誤差情報
                back_error.append(loss_der)
        else:
            # 隠れ層（次の層への入力の偏微分係数）
            back_error = grads_x[-1]  # 最後に追加された入力の勾配

        node_sums = cached_sums[layer_i - SKIP_INPUT_LAYER]
        for node_i, node_sum in enumerate(node_sums):  # 各ノードを処理

            # （2）活性化関数を偏微分
            if is_output_layer:
                # 出力層（恒等関数の微分）
                active_der = identity_der(node_sum)
            else:
                # 隠れ層（シグモイド関数の微分）
                active_der = sigmoid_der(node_sum)

            # （3）線形和を重み／バイアス／入力で偏微分
            w = weights[layer_i - SKIP_INPUT_LAYER][node_i]
            b = biases[layer_i - SKIP_INPUT_LAYER][node_i]
            x = cached_outs[layer_i - PREV_LAYER]  # 前の層の出力＝今の層への入力
            sum_der_w = sum_der(x, w, b, with_respect_to='w')
            sum_der_b = sum_der(x, w, b, with_respect_to='b')
            sum_der_x = sum_der(x, w, b, with_respect_to='x')

            # （4）各重み／バイアス／各入力の勾配を計算
            delta = back_error[node_i] * active_der

            # バイアスは1つだけ
            grad_b = delta * sum_der_b
            layer_grads_b.append(grad_b)

            # 重みと入力は前の層のノードの数だけある
            node_grads_w = []
            for x_i, (each_dw, each_dx) in enumerate(zip(sum_der_w, sum_der_x)):
                # 重みは個別に取得する
                grad_w = delta * each_dw
                node_grads_w.append(grad_w)

                # 入力は各ノードから前のノードに接続する全ての入力を合計する
                # （※重み視点と入力視点ではエッジの並び方が違うので注意）
                grad_x = delta * each_dx
                if node_i == 0:
                    # 最初に、入力の勾配を作成
                    layer_grads_x.append(grad_x)
                else:
                    # その後は、その入力の勾配に合計していく
                    layer_grads_x[x_i] += grad_x
            layer_grads_w.append(node_grads_w)

        # 層ごとの勾配を、ネットワーク全体用のリストに格納
        grads_w.append(layer_grads_w)
        grads_b.append(layer_grads_b)
        grads_x.append(layer_grads_x)

    # 保持しておいた各勾配（※逆順で追加したので反転が必要）を戻り値で返す
    grads_w.reverse()
    grads_b.reverse()
    return (grads_w, grads_b)  # grads_xは最適化で不要なので返していない

######## DNN22 ########
## 逆伝播の実行例 ##
x = [0.05, 0.1]
layers = [2, 2, 2]
weights = [
    [[0.15, 0.2], [0.25, 0.3]],
    [[0.4, 0.45], [0.5,0.55]]
]
biases = [[0.35, 0.35], [0.6, 0.6]]
model = (layers, weights, biases)
y_true = [0.01, 0.99]

print("Input data")
print(f'x={x}')
print(f'y_true={y_true}')

# （1）順伝播の実行例
y_pred, cached_outs, cached_sums = forward_prop(*model, x, cache_mode=True)
print("\nForward propagation")
print(f'y_pred={y_pred}')
print(f'cached_outs={cached_outs}')
print(f'cached_sums={cached_sums}')
# 出力例：
# y_pred=[1.10590596705977, 1.2249214040964653]
# cached_outs=[[0.05, 0.1], [0.5932699921071872, 0.596884378259767], [1.10590596705977, 1.2249214040964653]]
# cached_sums=[[0.3775, 0.39249999999999996], [1.10590596705977, 1.2249214040964653]]

# （2）逆伝播の実行例
grads_w, grads_b = back_prop(*model, y_true, cached_outs, cached_sums)
print("\nBack propagation")
print(f'grads_w={grads_w}')
print(f'grads_b={grads_b}')
# 出力例：
# grads_w=[[[0.006706025259285303, 0.013412050518570607], [0.007487461943833829, 0.014974923887667657]], [[0.6501681244277691, 0.6541291517796395], [0.13937181955411934, 0.1402209162240302]]]
# grads_b=[[0.13412050518570606, 0.14974923887667657], [1.09590596705977, 0.23492140409646534]]

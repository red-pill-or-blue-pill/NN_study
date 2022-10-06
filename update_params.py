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

######## DNN24 ########
## パラメーター更新の処理全体の実装（SGDの場合） ##
def update_params(layers, weights, biases, grads_w, grads_b, lr=0.1):
    """
    パラメーター（重みとバイアス）を更新する関数。
    - 引数：
    (layers, weights, biases)： モデルを指定する。
    grads_w： 重みの勾配。
    grads_b： バイアスの勾配。
    lr： 学習率（learning rate）。最適化を進める量を調整する。
    - 戻り値：
    新しい重みとバイアスを返す。
    """

    # ネットワーク全体で勾配を保持するためのリスト
    new_weights = [] # 重み
    new_biases = [] # バイアス

    SKIP_INPUT_LAYER = 1
    for layer_i, layer in enumerate(layers):  # 各層を処理
        if layer_i == 0:
            continue  # 入力層はスキップ

        # 層ごとで勾配を保持するためのリスト
        layer_w = []
        layer_b = []

        for node_i in range(layer):  # 層の中の各ノードを処理
            b = biases[layer_i - SKIP_INPUT_LAYER][node_i]
            grad_b = grads_b[layer_i - SKIP_INPUT_LAYER][node_i]
            b = b - lr * grad_b  # バイアスパラメーターの更新
            layer_b.append(b)

            node_weights = weights[layer_i - SKIP_INPUT_LAYER][node_i]
            node_w = []
            for each_w_i, w in enumerate(node_weights):
                grad_w = grads_w[layer_i - SKIP_INPUT_LAYER][node_i][each_w_i]
                w = w - lr * grad_w  # 重みパラメーターの更新
                node_w.append(w)
            layer_w.append(node_w)

        new_weights.append(layer_w)
        new_biases.append(layer_b)
    
    return (new_weights, new_biases)


######## DNN25 ########
## パラメーター更新の実行例 ##
print(" #25 ")
layers = [2, 2, 2]
weights = [
    [[0.15, 0.2], [0.25, 0.3]],
    [[0.4, 0.45], [0.5,0.55]]
]
biases = [[0.35, 0.35], [0.6, 0.6]]
model = (layers, weights, biases)

# 元の重み
print("=== Original weight & bias ===")
print(f'old-weights={weights}')
print(f'old-biases={biases}' )
# old-weights=[[[0.15, 0.2], [0.25, 0.3]], [[0.4, 0.45], [0.5, 0.55]]]
# old-biases=[[0.35, 0.35], [0.6, 0.6]]

# （1）順伝播の実行例
x = [0.05, 0.1]
y_pred, cached_outs, cached_sums = forward_prop(*model, x, cache_mode=True)

# （2）逆伝播の実行例
y_true = [0.01, 0.99]
grads_w, grads_b = back_prop(*model, y_true, cached_outs, cached_sums)
print("\n=== Back propagation ===")
print(f'grads_w={grads_w}')
print(f'grads_b={grads_b}')
# grads_w=[[[0.006706025259285303, 0.013412050518570607], [0.007487461943833829, 0.014974923887667657]], [[0.6501681244277691, 0.6541291517796395], [0.13937181955411934, 0.1402209162240302]]]
# grads_b=[[0.13412050518570606, 0.14974923887667657], [1.09590596705977, 0.23492140409646534]]

# （3）パラメーター更新の実行例
LEARNING_RATE = 0.1 # 学習率（lr）
weights, biases = update_params(*model, grads_w, grads_b, lr=LEARNING_RATE)

# 更新後の新しい重み
print("\n=== New weight & bias ===")
print(f'new-weights={weights}')
print(f'new-biases={biases}')
# new-weights=[[[0.14932939747407145, 0.19865879494814295], [0.2492512538056166, 0.2985025076112332]], [[0.3349831875572231, 0.3845870848220361], [0.48606281804458806, 0.5359779083775971]]]
# new-biases=[[0.3365879494814294, 0.33502507611233234], [0.490409403294023, 0.5765078595903534]]

# モデルの最適化
model = (layers, weights, biases)


######## DNN26 ########
## 訓練（最適化）処理全体を担う関数の実装 ##
print("\n\n #26 ")
import random

# 取りあえず仮で、空の関数を定義して、コードが実行できるようにしておく
def pre_optimize(model, x, y, data_i, last_i, batch_i, batch_size, acm_g, lr=0.1):
    " モデルを最適化する関数（子関数）。"
    loss = 0.1
    return model, loss, batch_i, acm_g

# ---ここまでは仮の実装。ここからが必要な実装---


def pre_train(model, x, y, batch_size=32, epochs=10, lr=0.1, verbose=10):
    """
    モデルの訓練を行う関数（親関数）。
    - 引数：
    model： モデルをタプル「(layers, weights, biases)」で指定する。
    x： 訓練データ（各データが行、各特徴量が列の、2次元リスト値）。
    y： 訓練ラベル（各データが行、各正解値が列の、2次元リスト値）。
    batch_size： バッチサイズ。何件のデータをまとめて処理するか。
    epochs： エポック数。全データ分で何回、訓練するか。
    lr： 学習率（learning rate）。最適化を進める量を調整する。
    verbose： 訓練状況を何エポックおきに出力するか。
    - 戻り値：
    損失値の履歴を返す。これを使って損失値の推移グラフが描ける。
    """
    loss_history = []  # 損失値の履歴

    data_size = len(y)  # 訓練データ数
    data_indexes = range(data_size)  # 訓練データのインデックス

    # 各エポックを処理
    for epoch_i in range(1, epochs + 1):  # 経過表示用に1スタート

        acm_loss = 0  # 損失値を蓄積（accumulate）していく

        # 訓練データのインデックスをシャッフル（ランダムサンプリング）
        random_indexes = random.sample(data_indexes, data_size)
        last_i = random_indexes[-1]  # 最後の訓練データのインデックス

        # 親関数で管理すべき変数
        acm_g = (None, None)  # 重み／バイアスの勾配を蓄積していくため
        batch_i = 0  # バッチ番号をインクリメントしていくため

        # 訓練データを1件1件処理していく
        for data_i in random_indexes:

            # 親子に分割したうちの子関数を呼び出す
            model, loss, batch_i, acm_g = pre_optimize(
                model, x, y, data_i, last_i, batch_i, batch_size, acm_g, lr)

            acm_loss += loss  # 損失値を蓄積

        # エポックごとに損失値を計算。今回の実装では「平均」する
        layers = model[0]  # レイヤー構造
        out_count = layers[-1]  # 出力層のノード数
        # 「訓練データ数（イテレーション数×バッチサイズ）×出力ノード数」で平均
        epoch_loss = acm_loss / (data_size * out_count)

        # 訓練状況を出力
        if verbose != 0 and \
            (epoch_i % verbose == 0 or epoch_i == 1 or epoch_i == EPOCHS):
            print(f'[Epoch {epoch_i}/{EPOCHS}] train_loss: {epoch_loss}')

        loss_history.append(epoch_loss)  # 損失値の履歴として保存

    return model, loss_history


# サンプル実行用の仮のモデルとデータ
layers = [2, 2, 2]
weights = [
    [[0.15, 0.2], [0.25, 0.3]],
    [[0.4, 0.45], [0.5,0.55]]
]
biases = [[0.35, 0.35], [0.6, 0.6]]
model = (layers, weights, biases)
x = [[0.05, 0.1]]
y = [[0.01, 0.99]]

# モデルを訓練する
BATCH_SIZE = 2  # バッチサイズ
EPOCHS = 1  # エポック数
LEARNING_RATE = 0.02 # 学習率（lr）
model, loss_history = pre_train(model, x, y, BATCH_SIZE, EPOCHS, LEARNING_RATE)
# 出力例：
# [Epoch 1/1] train_loss: 0.05



######## DNN27 ########
## 最適化処理を担う関数の実装（オンライン学習／ミニバッチ学習／バッチ学習に対応） ##
print("\n\n #27 ")

def train(model, x, y, batch_size=32, epochs=10, lr=0.1, verbose=10):
    """
    モデルの訓練を行う関数（親関数）。
    - 引数：
    model： モデルをタプル「(layers, weights, biases)」で指定する。
    x： 訓練データ（各データが行、各特徴量が列の、2次元リスト値）。
    y： 訓練ラベル（各データが行、各正解値が列の、2次元リスト値）。
    batch_size： バッチサイズ。何件のデータをまとめて処理するか。
    epochs： エポック数。全データ分で何回、訓練するか。
    lr： 学習率（learning rate）。最適化を進める量を調整する。
    verbose： 訓練状況を何エポックおきに出力するか。
    - 戻り値：
    損失値の履歴を返す。これを使って損失値の推移グラフが描ける。
    """
    loss_history = []  # 損失値の履歴

    data_size = len(y)  # 訓練データ数
    data_indexes = range(data_size)  # 訓練データのインデックス

    # 各エポックを処理
    for epoch_i in range(1, epochs + 1):  # 経過表示用に1スタート

        acm_loss = 0  # 損失値を蓄積（accumulate）していく

        # 訓練データのインデックスをシャッフル（ランダムサンプリング）
        random_indexes = random.sample(data_indexes, data_size)
        last_i = random_indexes[-1]  # 最後の訓練データのインデックス

        # 親関数で管理すべき変数
        acm_g = (None, None)  # 重み／バイアスの勾配を蓄積していくため
        batch_i = 0  # バッチ番号をインクリメントしていくため

        # 訓練データを1件1件処理していく
        for data_i in random_indexes:

            # 親子に分割したうちの子関数を呼び出す
            model, loss, batch_i, acm_g = optimize(
                model, x, y, data_i, last_i, batch_i, batch_size, acm_g, lr)

            acm_loss += loss  # 損失値を蓄積

        # エポックごとに損失値を計算。今回の実装では「平均」する
        layers = model[0]  # レイヤー構造
        out_count = layers[-1]  # 出力層のノード数
        # 「訓練データ数（イテレーション数×バッチサイズ）×出力ノード数」で平均
        epoch_loss = acm_loss / (data_size * out_count)

        # 訓練状況を出力
        if verbose != 0 and \
            (epoch_i % verbose == 0 or epoch_i == 1 or epoch_i == EPOCHS):
            print(f'[Epoch {epoch_i}/{EPOCHS}] train_loss: {epoch_loss}')

        loss_history.append(epoch_loss)  # 損失値の履歴として保存

    return model, loss_history



def accumulate(list1, list2):
    "2つのリストの値を足し算する関数。"
    new_list = []
    for item1, item2 in zip(list1, list2):
        if isinstance(item1, list):
            child_list = accumulate(item1, item2)
            new_list.append(child_list)
        else:
            new_list.append(item1 + item2)
    return new_list

def mean_element(list1, data_count):
    "1つのリストの値をデータ数で平均する関数。"
    new_list = []
    for item1 in list1:
        if isinstance(item1, list):
            child_list = mean_element(item1, data_count)
            new_list.append(child_list)
        else:
            new_list.append(item1 / data_count)
    return new_list


def optimize(model, x, y, data_i, last_i, batch_i, batch_size, acm_g, lr=0.1):
    "train()親関数から呼ばれる、最適化のための子関数。"

    layers = model[0]  # レイヤー構造
    each_x = x[data_i]  # 1件分の訓練データ
    y_true = y[data_i]  # 1件分の正解値

    # ステップ（1）順伝播
    y_pred, outs, sums = forward_prop(*model, each_x, cache_mode=True)

    # ステップ（2）逆伝播
    gw, gb = back_prop(*model, y_true, outs, sums)

    # 各勾配を蓄積（accumulate）していく
    if batch_i == 0:
        acm_gw = gw
        acm_gb = gb
    else:
        acm_gw = accumulate(acm_g[0], gw)
        acm_gb = accumulate(acm_g[1], gb)
    batch_i += 1  # バッチ番号をカウントアップ＝現在のバッチ数

    # 訓練状況を評価するために、損失値を取得
    loss = 0.0
    for output, target in zip(y_pred, y_true):
        loss += sseloss(output, target)

    # バッチサイズごとで後続の処理に進む
    if batch_i % BATCH_SIZE != 0 and data_i != last_i:
        return model, loss, batch_i, (acm_gw, acm_gb)  # バッチ内のデータごと

    layers = model[0]  # レイヤー構造
    out_count = layers[-1]  # 出力層のノード数

    # 平均二乗誤差なら平均する（損失関数によって異なる）
    grads_w = mean_element(acm_gw, batch_i * out_count)  # 「バッチサイズ ×
    grads_b = mean_element(acm_gb, batch_i * out_count)  # 　出力ノード数」で平均
    batch_i = 0  # バッチ番号を初期化して次のイテレーションに備える

    # ステップ（3）パラメーター（重みとバイアス）の更新
    weights, biases = update_params(*model, grads_w, grads_b, lr)

    # モデルをアップデート（＝最適化）
    model = (layers, weights, biases)

    return model, loss, batch_i, (acm_gw, acm_gb)  # イテレーションごと


layers = [2, 2, 2]
weights = [
    [[0.15, 0.2], [0.25, 0.3]],
    [[0.4, 0.45], [0.5,0.55]]
]
biases = [[0.35, 0.35], [0.6, 0.6]]
model = (layers, weights, biases)
x = [[0.05, 0.1]]
y = [[0.01, 0.99]]

# サンプル実行
model, loss_history = train(model, x, y, BATCH_SIZE, EPOCHS, LEARNING_RATE)
# 出力例：
# [Epoch 1/1] train_loss: 0.31404948868496607


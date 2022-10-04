# 取りあえず仮で、空の関数を定義して、コードが実行できるようにしておく
def summation(x,weights, bias):
    " 重み付き線形和の関数。"
    return 0.0

def sigmoid(x):
    " シグモイド関数。"
    return 0.0

def identity(x):
    " 恒等関数。"
    return 0.0


w = [0.0, 0.0]  # 重み（仮の値）
b = 0.0  # バイアス（仮の値）

next_x = x  # 訓練データをノードへの入力に使う

# ---ここまでは仮の実装。ここからが必要な実装---

# 1つのノードの処理（1）： 重み付き線形和
node_sum = summation(next_x, w, b)

# 1つのノードの処理（2）： 活性化関数
is_hidden_layer = True
if is_hidden_layer:
    # 隠れ層（シグモイド関数）
    node_out = sigmoid(node_sum)
else:
    # 出力層（恒等関数）
    node_out = identity(node_sum)

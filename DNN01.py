# 取りあえず仮で、空の関数を定義して、コードが実行できるようにしておく
def forward_prop(cache_mode=False):
    " 順伝播を行う関数。"
	return None, None, None

y_true = [1.0]  # 正解値
def back_prop(y_true, cached_outs, cached_sums):
    " 逆伝播を行う関数。"
	return None, None

LEARNING_RATE = 0.1 # 学習率（lr）
def update_params(grads_w, grads_b, lr=0.1):
    " パラメーター（重みとバイアス）を更新する関数。"
	return None, None

# ---ここまでは仮の実装。ここからが必要な実装---

# 訓練処理
y_pred, cached_outs, cached_sums = forward_prop(cache_mode=True)  # （1）
grads_w, grads_b = back_prop(y_true, cached_outs, cached_sums)  # （2）
weights, biases = update_params(grads_w, grads_b, LEARNING_RATE)  # （3）

print(f'予測値：{y_pred}')  # 予測値： None
print(f'正解値：{y_true}')  # 正解値： [1.0]


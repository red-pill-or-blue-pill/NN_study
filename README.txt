https://atmarkit.itmedia.co.jp/ait/series/15783/

TensorFlow 2＋Keras（tf.keras）入門
機械学習の勉強はここから始めてみよう。ディープラーニングの基盤技術であるニューラルネットワーク（NN）を、知識ゼロの状態から概略を押さえつつ実装。さらにCNNやRNNも同様に学ぶ。これらの実装例を通して、TensorFlow 2とKerasにも習熟する連載。

tf.keras-standard_nn.py
-https://atmarkit.itmedia.co.jp/ait/articles/1912/16/news026_3.html
-基本的なニューラルネットの説明とtensorflowでの計算の実例
-内容としてはブラウザ上のplayground-dataでのデモをしつつ、手元でpythonを動かして
-パラメータの設定方法などを解説　
-プログラムはテストデータを作成し、学習させ、結果を出力
-こちらも参照
-https://colab.research.google.com/github/DeepInsider/playground-data/blob/master/docs/articles/tf2_keras_neuralnetwork.ipynb?pli=1#scrollTo=Trk7kOFowxiQ

forward_prop.py
-https://atmarkit.itmedia.co.jp/ait/articles/2202/09/news027.html
-ニューラルネットの勉強用のソース
-順伝搬の例題でノードのweightを手作業で設定した上で
-入力を入れると途中の加算結果(cache)と結果(prep)を返す
-model 入力層：２、隠れ層１：３、出力層：１
-model2 入力層：２、隠れ層１：３、隠れ層２：２、出力層：１

back_prop.py
-https://atmarkit.itmedia.co.jp/ait/articles/2202/16/news032.html
-ニューラルネットの勉強用のソース
-逆伝搬の例題で、まずはノードのweightを手作業で設定した上で
-入力を入れると途中の加算結果(cache)と結果(prep)を返す
-（forward_prop.pyと一緒）
-その上で逆伝搬の計算を行い、重みとバイアスの計算を行う

update_params.py
-https://atmarkit.itmedia.co.jp/ait/articles/2202/28/news030.html
-ニューラルネットの勉強用のソース
-パラメータのアップデートのため順伝搬・逆伝搬をした上で
-算出した重み・バイアスでtrain関数で調整を行う
-optimize関数は最適化のための関数でtrainから呼ばれる
-最終的にはtrain関数を実行さえすれば、モデルと損失値を返す

solve_regression.py
-https://atmarkit.itmedia.co.jp/ait/articles/2202/28/news030.html
-ニューラルネットの勉強用のソース
-最終的に回帰問題を解く例題
-デモ用に訓練用データを作りtrainで処理された結果の
-model, loss_historyのうちloss_historyをプロットする
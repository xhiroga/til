import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():

    print (type(mnist)) # これ何型？

    x = tf.placeholder(tf.float32, [None, 784])
    # シェイプは配列で表す

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # (コスト定義)交差エントロピーを定義する
    y_ = tf.placeholder(tf.float32, [None,10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # y_は0or1, つまり1のはずのものを0.3Xとかで予想すると、ログは-XXXXとか巨大になり、ダメージがでかいようだ。
    # reduce_mean: 平均化によってランクをひとつ削減できるから？

    # 最小化のための関数を作成
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 複数回ランする準備
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print ("モデルによる予想 -> 誤差を重み付けに反映、を1000回繰り返す")
    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      if (i%100==0):
          print (str(i) + "周目の正答率...")
          scoring(x, y, y_, sess)

    print ("正誤判定 -> 平均正答率を算出")
    scoring(x, y, y_, sess)


def scoring(x, y, y_, sess):
    # 正誤判定
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # このtf.argmaxが便利。[0.1,0.75,0.15]とかを"2"にしてくれる（一番数字の高いのを返す）

    # 平均を計算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
    main()


"""
あとで質問すること:
1.トレーニングした結果はどうやって保存するのか？
（訓練済みデータでAPIを作りたい時など...jsonにしてファイルで保存する？）

2.inputデータをローカルから読み出す箇所、コピペで実装してよくわからない型の変数に代入したけど、
これは一体何型なの？

3.ハッカソンに活かしたい、そのへんお話ししたいです

参考URL
http://qiita.com/KojiOhki/items/ff6ae04d6cf02f1b6edf

"""

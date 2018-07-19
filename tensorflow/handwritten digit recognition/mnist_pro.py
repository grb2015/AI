# ref http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
import input_data
import tensorflow as tf
import os

'''# use gpu '''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
config.gpu_options.allow_growth = True


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# sess = tf.InteractiveSession()
sess = tf.InteractiveSession(config = config)  # use gpu

x = tf.placeholder("float", shape=[None, 784]) ## 待定数量的是图片
y_ = tf.placeholder("float", shape=[None, 10]) ## 图片对应的Lable

W = tf.Variable(tf.zeros([784,10])) ## 参数 用var表示
b = tf.Variable(tf.zeros([10])) ## 参数 用var表示

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)   ## softmax回归
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    ## 定义损失函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) ## 定义训练方法(梯度下降)

for i in range(1000):   ## 每次取一个batch进行训练
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  ## 统计得到的是否准确的bool类型行向量

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) ## 将bool类型转为float然后求平均数,得到准确率

print( accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) )

## --------------- 使用卷积提高准确率到99%  -----------------------------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  ##truncated_normal是什么？
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## 卷积使用1步长（stride size）, 0边距（padding size)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
##1. 第一层卷积
##卷积在每个5x5的patch中算出32个特征: 5*5是Patch大小，1是输入通道数目，32是输出通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  #每一个输出通道(总32）都有一个对应的偏置量

## 我们把x变成一个4d向量，其第2、第3维对应图片的宽、高
## 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
x_image = tf.reshape(x, [-1,28,28,1])

## 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##2. 第二层卷积

W_conv2 = weight_variable([5, 5, 32, 64])  ## 显然这里的输入通道为第一层的输出通道(32),第二层的输出通道为64
b_conv2 = bias_variable([64])

## 同上，我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


## 3. 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])  ## 图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  ## 这里的h_pool2注意是第二层的变量 ，我们把池化层输出的张量reshape成一些向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) ##乘上权重矩阵，加上偏置，然后对其使用ReLU

## 为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder("float")  ## 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  ## 调用tf的dropout函数

## 4 .输出层
## 添加一个softmax层，就像前面的单层softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


## 训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   ## 损失函数还是一样的
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  ## ADAM优化器来做梯度最速下降 ,与mnist.py中不一样
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) ##一样
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  ## 一样
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)  ## batch每次随机取50，训练20000次
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={  ## eval的作用跟 sess.run(accuracy) 差不多
        x:batch[0], y_: batch[1], keep_prob: 1.0})   ## feed_dict中加入额外的参数keep_prob来控制dropout比例

    print( "step %d, training accuracy %g"%(i, train_accuracy) )
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) )
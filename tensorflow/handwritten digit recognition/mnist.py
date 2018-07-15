#ref http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
#    https://www.cnblogs.com/lizheng114/p/7439556.html
import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])   ## 得到待定数量的图片，后面会被batch_xs赋值?对。即100个X,每个X有28*28的像素
print("x = ",x)

W = tf.Variable(tf.zeros([784,10]))	## 系数，初始化为0
b = tf.Variable(tf.zeros([10]))     ## 偏移，初始化为0
print("W = ",W)
print("b = ",b)

y = tf.nn.softmax(tf.matmul(x,W) + b)  ## 训练模型，我理解这里的x代表的应该是一张图片，y是一个行向量(属于0,1...9的概率）

y_ = tf.placeholder("float", [None,10])  ## None 后面会给100 即100张图片，即100个行向量
print('y = ',y)
print('y_ = ',y_)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))  ### 定义损失函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) ## 定义梯度下降训练，使损失函数最小

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):  ## 开始训练，训练1000次，每次随机从训练集中取100张图片
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  ## 这里x: batch_xs即用x替换之前的占位符x 。y_同理
print("end for ")
## 这行代码会返回一个行向量，size为100 ，即每张图片是否预测准确了.bool 类型 [1,0,1...1]
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) ## tf.argmax(y,1) 返回y向量最大的那个值的下标
print("correct_prediction = ",correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  ## 将bool类型转为float并求平均值
print("train accuracy = ",accuracy)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  ## 用测试集进行测试，返回准确率
###########################对每一张图片，打印训练结果 #############################################################################
#for i in range(0, len(mnist.test.images)):
TRAIN_LEN = len(mnist.test.images)  ## TRAIN_LEN = 1W
SAMPLE_LEN = 10
## 从0~TRAIN_LEN 中随机取出SAMPLE_LEN个数
list_random_int = random.sample(range(0, TRAIN_LEN), SAMPLE_LEN) ## index = 1364 或 9430的图片识别不准
print("list_random_int = ",list_random_int)
sucess_count = 0
for i in list_random_int:  
    ## 为什么有时会退出
    result = sess.run(correct_prediction,
                    feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])})
    print("----------------演示识别第%d/%d张图片... "%(i,SAMPLE_LEN))
    #if  result:
    print("sess.run result = %s"%result)
    p_pred = sess.run(y, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])})
    p_acture = sess.run(y_, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])})
    print("预测的值是(概率向量):", p_pred)
    print("实际值是(概率向量):", p_acture)

    pred_num = sess.run(tf.argmax(p_pred,1))
    acture_num = sess.run(tf.argmax(p_acture,1))
    print("预测的值是(数字):",pred_num,'type(pred_num) = ',type(pred_num))
    print("实际值是(数字):", acture_num)
    if pred_num == acture_num :
      sucess_count += 1
      print("sucess_count = ",sucess_count)


    ## 从mnist.test.images放过画出图片
    one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
    pic_matrix = np.matrix(one_pic_arr, dtype="float")
    plt.imshow(pic_matrix)
    pylab.show()
  #else:
      #print("#### run correct_prediction error")
      #break
percent_sucess_rate = '{:.2%}'.format(sucess_count/SAMPLE_LEN)

print("随机%d张测试集图片预测准确率:%s" % (SAMPLE_LEN,percent_sucess_rate ))  ## must %s , not %d
    #break
Accuracy_rate_1w = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels})
print("%d张测试集图片预测准确率:%s"%(TRAIN_LEN,Accuracy_rate_1w)) ## must %s , not %d


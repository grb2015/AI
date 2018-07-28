#ref http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
#    https://www.cnblogs.com/lizheng114/p/7439556.html
## brief : use  softmax regression to  recognition handwritten digit
# TODO(rbguo)  we need to draw a picture and resize it into  28*28  , and then copy it to MNIST_data_myself/

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
def train_offical_dataset():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder("float", [None, 784])   ## 得到待定数量的图片，后面会被batch_xs赋值?对。即100个X,每个X有28*28的像素
    print("x = ",x)

    W = tf.Variable(tf.zeros([784,10]))	## 系数，初始化为0
    b = tf.Variable(tf.zeros([10]))     ## 偏移，初始化为0
    print("W = ",W)
    print("b = ",b)

    y = tf.nn.softmax(tf.matmul(x,W) + b)  ## 训练模型，这里的x是什么?
                            ## X是代表了多张图片，比如下面batch为100,则X就包含了100张图片的信息，即X=[100*784] 是一个100行，784列矩阵
                            ## 而W为[784*10]的矩阵，所以X*W = [100*784]*[784*10]=[100*10] 即100行，10列的矩阵。
                            ## 所以y也是[100*10]的矩阵:
                            ##      y11代表第1张图片是数字0的概率,y12代表第1张图片是数字1的概率,...
                            ##      y21代表第2张图片是数字0的概率...        
                   
    y_ = tf.placeholder("float", [None,10])  ## None 后面会给100 即100张图片，即100个行向量
    print('y = ',y)
    print('y_ = ',y_)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))  ### 定义损失函数 reduce_sum就是求和

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) ## 定义梯度下降训练，使损失函数最小

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    ## 这个训练的过程，其实就是改变W和b的权重的过程
    for i in range(1000):  ## 开始训练，训练1000次，每次随机从训练集中取100张图片
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  ## 这里x: batch_xs即用x替换之前的占位符x 。y_同理
        if i%100 == 0:
            print("i = ",i)
            print("w =",sess.run(W) )  ## 为什么W总是zero ?
            print("b =",sess.run(b) )
            print("\n\n")
    ## 这行代码会返回一个行向量， correct_prediction size为100 ，即每张图片是否预测准确了.bool 类型 [1,0,1...1]
    ## 疑问？这里的y和y_ 包含多少数据?
    ##   答： 实际上只是定义了correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)),下面对测试dataset
    ## 的操作，自然会给定y_,y : accuracy_rate = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    ##   所以 y_ 就是   y_: mnist.test.label

    #print("## y = ",sess.run(y))  ## 这里打印会报错，因为x 还没有填充
    #print("## y_ = ", sess.run(y_)) ## 同理
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) ## tf.argmax(y,1) 返回y向量最大的那个值的下标
    #print("correct_prediction = %s,len(correct_prediction) = %s"%(correct_prediction,len(correct_prediction)))
    print("correct_prediction = ",correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  ## 将bool类型转为float并求平均值
    ##print("train accuracy = ",accuracy)
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    ## 用测试集进行测试，返回准确率
    TRAIN_LEN = len(mnist.test.images)
    accuracy_rate = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("%d张测试集图片预测准确率:%s"%(TRAIN_LEN,accuracy_rate)) ## must %s , not %d
    print("acc =", accuracy_rate)
    return accuracy,x,y_,mnist,sess,y   ## 训练结束后，其实就是得到了最终W和b的值


###########################对每一张图片，打印训练结果 #############################################################################
def test_offfical_simple(x,y_,sess,y):
    MNIST_data_dir = "MNIST_data/"
    test_simple(MNIST_data_dir,x, y_, sess, y)

def test_myself_handwrite_digit(x,y_,sess,y):
    NIST_data_dir = "MNIST_data_myself/"
    test_simple(NIST_data_dir, x, y_, sess, y)

def test_simple(MNIST_data_dir,x, y_, sess, y):
    #for i in range(0, len(mnist.test.images)):
    mnist = input_data.read_data_sets(MNIST_data_dir, one_hot=True)

    TEST_IMAGE_LEN = len(mnist.test.images)  ## TRAIN_LEN = 1W
    SAMPLE_LEN = 10
    if (SAMPLE_LEN > TEST_IMAGE_LEN):
        SAMPLE_LEN = TEST_IMAGE_LEN
    ## 从0~TEST_IMAGE_LEN 中随机取出SAMPLE_LEN个数
    list_random_int = random.sample(range(0, TEST_IMAGE_LEN), SAMPLE_LEN) ## index = 1364 或 9430的图片识别不准
    print("list_random_int = ",list_random_int)
    sucess_count = 0
    for i,value in enumerate(list_random_int):
        ## 为什么有时会退出
        #result = sess.run(correct_prediction,
        #               feed_dict={x: np.array([mnist.test.images[value]]), y_: np.array([mnist.test.labels[value]])})
        print("----------------演示识别第%d/%d张图片... "%(i+1,SAMPLE_LEN))
        #if  result:
        #print("sess.run result = %s"%result)
        p_pred = sess.run(y, feed_dict={x: np.array([mnist.test.images[value]]), y_: np.array([mnist.test.labels[value]])})
        p_acture = sess.run(y_, feed_dict={x: np.array([mnist.test.images[value]]), y_: np.array([mnist.test.labels[value]])})
        print("预测的值是(概率向量):", p_pred)
        print("实际值是(概率向量):", p_acture)

        pred_num = sess.run(tf.argmax(p_pred,1))
        acture_num = sess.run(tf.argmax(p_acture,1))
        print("预测的值是(数字):",pred_num,'type(pred_num) = ',type(pred_num))
        print("实际值是(数字):", acture_num)
        if pred_num == acture_num :
          sucess_count += 1
          print("sucess_count = ",sucess_count)

        ## 从mnist.test.images反过来画出图片
        one_pic_arr = np.reshape(mnist.test.images[value], (28, 28))
        pic_matrix = np.matrix(one_pic_arr, dtype="float")
        plt.imshow(pic_matrix)
        pylab.show()
        '''
        print("show labse ...")
        one_pic_arr = np.reshape(mnist.test.labels[value], (1, 10))
        pic_matrix = np.matrix(one_pic_arr, dtype="float")
        plt.imshow(pic_matrix)
        pylab.show()
        '''
    percent_sucess_rate = '{:.2%}'.format(sucess_count/SAMPLE_LEN)

    print("随机%d张测试集图片预测准确率:%s" % (SAMPLE_LEN,percent_sucess_rate ))  ## must %s , not %d

def test(accuracy,x,y_,sess):  ### 这里用到了accuracy也是可以的，注意和test_simpleq区分
    print("----------------------------------------")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #x = tf.placeholder("float", [None, 784])  #在这里加竟然不行,必须参数传进来，为什么?
    #y_ = tf.placeholder("float", [None, 10])

    #init = tf.initialize_all_variables()      ## 自己加sess也不行
    #sess = tf.Session()
    #sess.run(init)
    TRAIN_LEN = len(mnist.test.images)
    #accuracy_rate = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    accuracy_rate = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("acc =",accuracy_rate)
    print("%d张测试集图片预测准确率:%s" % (TRAIN_LEN, accuracy_rate))  ## must %s , not %d
if __name__ == '__main__':
    accuracy,x,y_,mnist,sess,y = train_offical_dataset()
    #test(accuracy,x,y_,sess)
    test_offfical_simple(x,y_,sess,y)
    test_myself_handwrite_digit(x,y_,sess,y)


import tensorflow as tf

print("----1.构建图 op  -----")
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)


print("matrix1 = ",matrix1)
## output: Tensor("Const:0", shape=(1, 2), dtype=float32) 可见是一个Tensor对象,shape是行列数(1行2列)
print("matrix2 = ",matrix2)
print("product = ",product)

print("----2.在一个会话中启动图 session -----")
# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
'''
with tf.Session() as sess:
  result = sess.run([product])
  print result
'''

print("----3.Variables (计数器，如何打印变量) -----")
# 1.创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 2.创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 3.启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 4.启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print( sess.run(state) )
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print( 'state = ',sess.run(state),'one = ',sess.run(one) )

# 输出:
# 0
# 1
# 2
# 3
print("----4.Fetch (我们只取回了单个节点 state, 但是你也可以取回多个 tensor)-----")
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
# mul = tf.mul(input1, intermed)   ## tf.mul is removed in new version
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])  ## 返回多个变量
  print(result)

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

print("----5.Feed 占位符(占的位可以是input1 constant,或者矩阵 )-----")
#input1 = tf.placeholder(tf.types.float32)  #
#input2 = tf.placeholder(tf.types.float32)
input1 = tf.placeholder(tf.float32) ## new version ,use tf.float32 占位符，表示一个后面填充的变量
input2 = tf.placeholder(tf.float32)## 占位符，表示一个后面填充的变量

output = tf.multiply(input1, input2)

with tf.Session() as sess:
  print ( sess.run([output], feed_dict={input1:[7.], input2:[2.]}) )## 填充变量

# 输出:
# [array([ 14.], dtype=float32)]
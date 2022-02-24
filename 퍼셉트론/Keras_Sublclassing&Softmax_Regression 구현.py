import tensorflow as tf

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
# 28 * 28 형태의 이미지를 784차원으로 flattening
x_train,x_test = x_train.reshape([-1,784]),x_test.reshape([-1,784])
#데이터 값을 [0,1]로 정규화
x_train,x_test = x_train/255 , x_test/255
# 레이블 데이터에 ohe 적용
y_train,y_test = tf.one_hot(y_train,depth = 10),tf.one_hot(y_test,depth = 10)

# tf.data API를 이용해서 데이터를 섞고 Mini Batch 형태로 가져와보기
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)) # 데이터를 쪼갠후
# repeat() : 전체 epoch의 반복횟루르 지정하는 옵션, 인자가없으면, 무한한  epoch만큼
# 여기서 repeat()을 뺀다면, 600개의 데이터 1회만 학습하므로 안된다.
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data)  #600개의 데이터를 iterable 하게

W = tf.Variable(tf.zeros(shape = [784,10])) # 784개의 테스트 열이있고, 10개의 원핫인코딩때문
b =tf.Variable(tf.zeros(shape = [10]))

@tf.function
def softmax_regression(x) :
    logits = tf.matmul(x,W) + b
    return tf.nn.softmax(logits)


@tf.function
def cross_entropy_loss(y_pred,y) :
    return tf.reduce_mean(-tf.reduce_sum(y *tf.math.log(y_pred),axis = [1]))

@tf.function
def compute_accuracy(y_pred,y) :
    #얼마나 같나?
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return accuracy

optimizer = tf.optimizers.SGD(0.5)

#최적화를 위한 function 정의
@tf.function
def train_step(x,y) :
    with tf.GradientTape() as tape :
        y_pred = softmax_regression(x)
        loss = cross_entropy_loss(y_pred,y)
    gradients = tape.gradient(loss,[W,b])
    optimizer.apply_gradients(zip(gradients,[W,b]))

for i in range(1000) :
    batch_xs,batch_ys = next(train_data_iter)
    train_step(batch_xs,batch_ys)

print('정확도 :%f'% compute_accuracy(softmax_regression(x_test),y_test))
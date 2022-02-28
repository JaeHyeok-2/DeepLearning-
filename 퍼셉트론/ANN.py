import tensorflow as tf

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
# 28 * 28 형태의 이미지를 784차원으로
x_train,x_test = x_train.reshape([-1,784]),x_test.reshape(-1,784)

x_train,x_test = x_train/255 ,x_test/255

y_train,y_test = tf.one_hot(y_train,depth = 10 ), tf.one_hot(y_test,depth=10)

learning_rate = 0.001
num_epoches = 30 # 학습횟수
batch_size = 256
display_step = 1 # 손실함수 출력 주기
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10
"""

784개의 데이터 -> 256개의 h1 - >256 h2 -> 10개의 결과 ?

"""
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.shuffle(60000).batch(batch_size)

class ANN(object) :
    #ANN 모델 생성
    def __init__(self):
        # w1 : (55000 x 784)개의 데이터
        self.W1 = tf.Variable(tf.random.normal(shape = [input_size,hidden1_size])) # (784,256)
        self.b1 = tf.Variable(tf.random.normal(shape = [hidden1_size])) # (256,1)
        self.W2 = tf.Variable(tf.random.normal(shape = [hidden1_size,hidden2_size])) # (256,256)
        self.b2 = tf.Variable(tf.random.normal(shape = [hidden2_size])) # (256,1)
        self.W_output = tf.Variable(tf.random.normal(shape = [hidden2_size, output_size])) #(256,10)
        self.b_output = tf.variable(tf.random.normal(shape = [output_size])) #(10,1)
        # b는 1개가 나오고 w 가 10개

    # call 함수:  함수를 호출하듯이, 클래스의 객체도 호출하도록

    def __call__(self,x):
        # x *W1 + b1
        H1_output = tf.nn.relu(tf.matmul(x,self.W1)+self.b1)
        #((x *w1 +b1) *W2) + b2
        H2_output = tf.nn.relu(tf.matmul(H1_output,self.W2) + self.b2)
        # (((x *w1 +b1) *W2) + b2) * W_output + b_output
        logits = tf.matmul(H2_output,self.W_output) + self.b_output

        return logits


@tf.function
def cross_entropy_loss(logits,y) :
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels =y))

#최적화를 위한 Adam 옵티마이저를 정의합니다
optimizer = tf.optimizers.Adam(learning_rate)

# 최적화를 위한 function을 정의합니다
def train_step(model,x,y) :
    with tf.GradientTape() as tape :
        y_pred = model(x) #
        loss = cross_entropy_loss(y_pred,y)
        #vars() : 객체의 __dict__ 속성 반환
    # 후진 방식 자동 미분을 사용해 테이프에 기록된 연산의 그래디언트를 계산
    gradients = tape.gradient(loss,vars(model).values())
    optimizer.apply_gradients(zip(gradients,vars(model).values()))


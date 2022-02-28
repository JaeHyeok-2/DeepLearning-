import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#MNIST 데이터 다운
(x_train,y_train),(x_test,y_test ) = tf.keras.datasets.mnist.load_data()
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
#28 *28형태의 이미지를 784차원의 flattening
x_train,x_test = x_train.reshape([-1,784]),x_test.reshape([-1,784])
x_train,x_test = x_train/255, x_test/255

learning_rate = 0.02
training_epoches = 50 # 반복
batch_size =256
display_step = 1
examples_to_show = 10
input_size = 784
hidden_size1 =256
hidden_size2 = 128

train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.shuffle(60000).batch(batch_size)

class AutoEncoder(object) :
    def __init__(self):
        #Encoding = 784 -> 256 ->128
        self.W1 = tf.Variable(tf.random.normal(shape = [input_size,hidden_size1]))
        self.b1 = tf.Variable(tf.random.normal(shape= [hidden_size1]))
        self.W2 = tf.Variable(tf.random.normal(shape= [hidden_size1,hidden_size2]))
        self.b2 = tf.Variable(tf.random.normal(shape= [hidden_size2]))

        # Decoding => 128->256->784 대칭
        self.W3 = tf.Variable(tf.random.normal(shape = [hidden_size2,hidden_size1]))
        self.b3 = tf.Variable(tf.random.normal(shape = [hidden_size1]))
        self.W4 = tf.Variable(tf.random.normal(shape = [hidden_size1,input_size]))
        self.b4 = tf.Variable(tf.random.normal(shape = [input_size]))

    def __call__(self,x):
        H1_output = tf.nn.sigmoid(tf.matmul(x,self.W1) + self.b1)
        H2_output = tf.nn.sigmoid(tf.matmul(H1_output,self.W2) + self.b2)
        H3_output = tf.nn.sigmoid(tf.matmul(H2_output,self.W3) + self.b3)
        reconstructed_x = tf.nn.sigmoid(tf.matmul(H3_output,self.W4) + self.b4)

        return reconstructed_x

#mse 함수 생성
def mse_loss(y_pred,y_true) :
    return tf.reduce_mean(tf.pow(y_pred- y_true,2))

#옵티마이저
optimizer =tf.optimizers.RMSprop(learning_rate)

#최적화 함수
def train_step(model,x) :
    y_true = x
    with tf.GradientTape() as tape :
       y_pred = model(x)
       loss =  mse_loss(y_pred,y_true)
    gradients = tape.gradient(loss,vars(model).values())
    optimizer.apply_gradients(zip(gradients,vars(model).values()))

AutoEncoder_model = AutoEncoder()

for epoch in range(training_epoches) :
   #모든 배치들에 대해서 최적화를 수행
   #AutoEncoder 는 비지도 학습이므로 labels 값이 필요 x
   for batch_x in train_data :
       _,current_loss = train_step(AutoEncoder_model,batch_x),\
                        mse_loss(AutoEncoder_model(batch_x),batch_x)

   if epoch % display_step == 0 :
       print('반복(epoch) : %d, 손실함수(Loss) : %f' % ((epoch+1),current_loss))


reconstructed_result = AutoEncoder_model(x_test[:examples_to_show])

f,a = plt.subplots(2,10,figsize=(10,2))

for i in range(examples_to_show) :
    a[0][i].imshow(np.reshape(x_test[i],(28,28)))
    a[1][i].imshow(np.reshape(reconstructed_result[i],(28,28)))
f.savefig('reconstructed_mnist_image.png')
f.show()
plt.draw()
plt.waitforbuttonpress()


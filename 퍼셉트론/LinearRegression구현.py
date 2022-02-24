import tensorflow as tf

# Y = W * x+ b  에서 임의의 기울기 값 생성
W = tf.Variable(tf.random.normal(shape= [1]))
b = tf.Variable(tf.random.normal(shape =[1]))

# 손실함수 생성
@tf.function
def linear_model(x) :
    return W * x + b

#MSE 손실함수 mean(y`-y)^2
@tf.function
def mse_loss(y_pred,y) :
    return tf.reduce_mean(tf.square(y_pred-y))

#옵티마이저 (Optimizer)는 손실 함수을 통해 얻은 손실값으로부터 모델을 업데이트하는 방식을 의미합니다.
#TensorFlow는 SGD, Adam, RMSprop과 같은 다양한 종류의 옵티마이저를 제공합니다.
optimizer = tf.optimizers.SGD(0.01) # 0.01 =learning_rate 값

# 최적화를 위한 함수 정의
@tf.function
def train_step(x,y ) :
    # GradientTape는 자동미분계산을 해주고, 그결과값을 tape에 자동으로 저장 (with 구문 내에 실행된 값들)
    with tf.GradientTape() as tape :
        # train데이터 x에 대한 y의 예측값
        y_pred = linear_model(x)
        # 예측값과 실제값을 통한 mse 오차 방식구하기
        loss = mse_loss(y_pred,y)
    # tape내에 있던 그라디언트 값들 tape.gradient(loss,(모델의 trainable_variables))
    gradients = tape.gradient(loss,[W,b])
    # 오차 역전파(Backpropagation) -> weight를 업데이트
    optimizer.apply_gradients(zip(gradients,[W,b]))

x_train = [1,2,3,4]
y_train = [2,4,6,8]

#1000번의 경사하강법 수행
for i in range(1000) :
    train_step(x_train,y_train)

#테스트를 위한 입력값을 준비
x_test = [3.5,5,5.5,6]

#테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)확인
print(linear_model(x_test).numpy())
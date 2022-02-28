import tensorflow as tf

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test = x_train.astype('float32'),x_test.astype('float32')
x_train,x_test = x_train/255, x_test/255

y_train_one_hot = tf.squeeze(tf.one_hot(y_train,10),axis= 1 )
y_test_one_hot = tf.squeeze(tf.one_hot(y_test,10),axis = 1)

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test_one_hot))
test_data  =test_data.batch(1000)
test_data_iter = iter(test_data)

class CNN(tf.keras.Model) :
    def __init__(self):
        super(CNN,self).__init__()
        # 첫번째 convolutional layer - 하나의 RGB 이미지를 64개의 특징들으로 맵핑
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 5, strides = 1
                                                   , padding = 'same', activation  ='relu')
        # MaxPool 로
        self.pool_layer_1 = tf.keras.layer.MaxPool2D(pool_size = (3,3),strides = 2)
        # 두번째 convolutional layer - 64개의 특징을 64개의 특징들로 맵핑
        self.conv_layer_2=  tf.keras.layers.Conv2D(filters = 64, kernel_size =5 , strides = 1
                                                   , padding = 'same',activation = 'relu')
        self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size = (3,3),strides = 2)
        # 세번째 convolutional layer
        self.conv_layer_3 = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides =1, padding = 'same',
                                                   activation = 'relu')
        # 네번째 convolutional layer
        self.conv_layer_4 = tf.keras.layers.Conv2D(filters = 128,kernel_size = 3, strides = 1, padding ='same',
                                                   activation = ' relu')
        # 다섯번째 convolutional layer
        self.conv_layere_5 = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides =3, padding = 'same',
                                                    activation = 'relu')

        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(384,activation = 'relu')
        self.dropout = tf.keras.layers.Dropout(0.2)

        # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스 -airplane,automible,bird .. 로 맵핑합니다.
        self.output_layer = tf.keras.layers.Dense(10,activation = None)

    def call(self,x,is_training):

        #입력 이미지
        h_conv1 = self.conv_layer_1(x)
        h_pool1 = self.pool_layer_1(h_conv1)
        h_conv2 = self.conv_layer_2(h_pool1)
        h_pool2 = self.pool_layer_2(h_conv2)
        h_conv3 = self.conv_layer_3(h_pool2)
        h_conv4 = self.conv_layer_4(h_conv3)
        h_conv5 =self.conv_layer_5(h_conv4)
        h_conv5_flat = self.flatten_layer(h_conv5)
        h_fc1 = self.fc_layer_1(h_conv5_flat)


############
# Step1
# 辨識資料處裡
############

# 匯入模組
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

# 獲取mnist資料
from keras.datasets import mnist
(x_train_image, y_train_label),\
(x_test_image, y_test_label) = mnist.load_data()

# 將影像特徵值使用reshape轉換
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(60000, 784).astype('float32')

# 將影像特徵值做標準化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

# 將數字的真實數值做One-hot encoding轉換
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

###########
# Step2
# 建立模型
###########

# 匯入所需要之模組
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 建立Sequential模型
model = Sequential()

# 建立輸入層與隱藏層
#
# units隱藏層神經元個數
# input_dim輸入層神經元個數
# activation 設定激活函數
model.add(Dense(units = 256,
				input_dim = 784,
				kernel_initializer = 'normal',
				activation = 'relu'))
model.add(Dropout)

# 建立輸出層
model.add(Dense(units = 10,
				kernel_initializer = 'normal',
				activation = 'softmax'))

##############
# Step3
# 開始訓練
##############

# 定義訓練模式
# loss設定損失函數
# optimizer設定優化方法
# metrics設定評估模型方式
model.compile(loss = 'categorical_crossentropy',
			  optimizer = 'adam',
			  metrics = ['accuracy'])

# 訓練
# validation_split設定驗證資料之比率
# epochs 設定週期次數
# batch_size 設定每一批次多少筆資料
train_history = model.fit(x = x_Train_normalize,
						  y = y_Train_OneHot,
						  validation_split = 0.2,
						  epochs = 10,
						  batch_size = 200,
						  verbose = 2)

# 建立函數來顯示訓練過程
import matplotlib.pylot as plt
def show_train_history(train_history, train, validation):
	plt.plot(train_history, history[train])
	plt.plot(train_history, history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

# 繪製accuracy準確率執行結果
show_train_history(train_history, 'acc', 'val_acc')

# 繪製loss誤差執行結果
show_train_history(train_history, 'loss', 'val_loss')


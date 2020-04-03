import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fasion_mnist = keras.datasets.fashion_mnist

"""
load_data() 함수를 호출하면 4개의 numpy 배열이 반환된다.

train_images 와 train_labels 배열은 모델 학습에 사용되는 훈련세트
test_images, test_labels 배열은 모델 테스트에 사용되는 테스트 세트 
"""
(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()


"""
이미지는 28x28 크기의 넘파이 배열이고 픽셀 값은 0과 255 사이입니다. 레이블(label)은 0에서 9까지의 정수 배열입니다. 이 값은 이미지에 있는 옷의 클래스(class)를 나타냅니다:

레이블	클래스
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
각 이미지는 하나의 레이블에 매핑되어 있습니다. 데이터셋에 클래스 이름이 들어있지 않기 때문에 나중에 이미지를 출력할 때 사용하기 위해 별도의 변수를 만들어 저장합니다:

"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
모델을 훈련하기 전에 데이터셋 구조를 살펴보죠. 
다음 코드는 훈련 세트에 60,000개의 이미지가 있다는 것을 보여줍니다. 
각 이미지는 28x28 픽셀로 표현됩니다:
"""
train_images.shape


"""
픽셀값 범위가 0~255 사이라는 것을 알 수 있다.
"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


"""
신경말 모델에 주입하기 전에 이 값의 범위를 0~1 사이로 조정. 
훈련세트와 테스트 세트를 동일한 방식으로 전처리 하는것이 중요
"""

train_images = train_images / 255.0
test_images = test_images / 225.0


"""
데이터 포맷이 올바른지 확인가능.
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


"""
모델구성 
신경망 모델을 만들려면 모델의 층을 구성한 다음 모델을 컴파일 해준다.
"""

"""
층 설정
신경망의 기본 구성요소는 층(layer)이다. 
층은 주입된 데이터에서 표현을 추출한다. (문제를 해결하는데 더 의미있는 표현이 추출되기 위함을 기대)

대부분 딥러닝은 간단한 층을 연결하여 구성됩니다. tf.keras.layers.Dense와 같은 층들의 가중치(parameter)는 훈련하는 동안 학습됩니다.

첫 번째 층인 tf.keras.layers.Flatten은 2차원 배열(28 x 28 픽셀)의 이미지 포맷을 28 * 28 = 784 픽셀의 1차원 배열로 변환
이미지에 있는 픽셀의 행을 펼쳐서 일렬로 늘립니다. 이 층에는 학습되는 가중치가 없고 데이터를 변환하기만 합니다.

픽셀을 펼친 후 두 개의 tf.keras.layer.Dense 층이 연속되어 연결된다. 이 층을 밀집연결 또는 완전 연결층이라고 부른다.
첫 번째 Dense 층은 128개의 노드(또는 뉴런)를 가진다. 두 번째 (마지막) 층은 10개의 노드의 소프트맥스(softmax) 층이다.
이 층은 10개의 확률을 반환하고 반환된 값의 전체 합은 1이다. 각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력한다.

"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

"""모델 컴파일
손실 함수(Loss function)-훈련 하는 동안 모델의 오차를 측정합니다. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 합니다.
옵티마이저(Optimizer)-데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정합니다.
지표(Metrics)-훈련 단계와 테스트 단계를 모니터링하기 위해 사용합니다. 다음 예에서는 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.
"""

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


"""모델훈련
훈련 데이터를 모델에 주입합니다-이 예에서는 train_images와 train_labels 배열입니다.
모델이 이미지와 레이블을 매핑하는 방법을 배웁니다.
테스트 세트에 대한 모델의 예측을 만듭니다-이 예에서는 test_images 배열입니다. 이 예측이 test_labels 배열의 레이블과 맞는지 확인합니다.
"""
model.fit(train_images, train_labels, epochs=5)


"""
정확도 평가

테스트 세트의 정확도가 훈련 세트의 정확도보다 조금 낮습니다. 
훈련 세트의 정확도와 테스트 세트의 정확도 사이의 차이는 과대적합(overfitting) 때문입니다. 
과대적합은 머신러닝 모델이 훈련 데이터보다 새로운 데이터에서 성능이 낮아지는 현상을 말합니다.
"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)



"""
예측만들기 
"""

predictions = model.predict(test_images)

""" predictions[0] 실행시 10개의 숫자배열로 나타난다.
이 값은 10개의 옷 품목에 상응하는 모델의 신뢰도(confidence)를 나타난다.

array([1.6537431e-05, 8.6610027e-07, 1.9992506e-06, 9.1384734e-08,
       1.2081074e-06, 1.7686512e-02, 9.6968342e-06, 1.6786481e-01,
       2.6662360e-04, 8.1415164e-01], dtype=float32)
"""

"""
가장 높은 신뢰도를 가진 레이블을 찾아보기 
"""

s = np.argmax(predictions[0])
b = test_labels[0]

"""
모델은 이 이미지가 앵클 부츠(class_name[9])라고 가장 확신하고 있습니다. 
실제로도 이 값이 맞는 지 확인하기.
"""
print(s, b)



"""
10개 클래스에 대한 예측을 모두 그래프로 표현하기
"""

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


#테스트 세트에서 이미지 하나를 선택
img = test_images[0]
print(img.shape)

"""tf.keras 모델은 한 번에 샘플의 묶음 또는 배치(batch)로 예측을 만드는데 최적화되어 있습니다. 
하나의 이미지를 사용할 때에도 2차원 배열로 만들어야 합니다:
"""
# 이미지 하나만 사용할 때도 배치에 추가하기.
img = (np.expand_dims(img, 0))

#이미지 예측
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#model.predict는 2차원 넘파이 배열을 반환하므로 첫 번째 이미지의 예측을 선택합니다
np.argmax(predictions_single[0])
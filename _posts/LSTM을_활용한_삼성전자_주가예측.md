---
title: 최종 결과물 코드
tags: P15
categories: bigdata
---
# LSTM을 활용한 주가 예측 모델

## import


~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

%matplotlib inline
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'NanumGothic'
```

## 데이터 (FinanceDataReader)


```python
import FinanceDataReader as fdr
```


```python
# 삼성전자(005930) 전체 (1996-11-05 ~ 현재)
samsung = fdr.DataReader('005930')
```


```python
samsung.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-12-18</th>
      <td>73300</td>
      <td>73700</td>
      <td>73000</td>
      <td>73000</td>
      <td>17613029</td>
      <td>-0.004093</td>
    </tr>
    <tr>
      <th>2020-12-21</th>
      <td>73100</td>
      <td>73400</td>
      <td>72000</td>
      <td>73000</td>
      <td>20367355</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-12-22</th>
      <td>72500</td>
      <td>73200</td>
      <td>72100</td>
      <td>72300</td>
      <td>16304910</td>
      <td>-0.009589</td>
    </tr>
    <tr>
      <th>2020-12-23</th>
      <td>72400</td>
      <td>74000</td>
      <td>72300</td>
      <td>73900</td>
      <td>19411326</td>
      <td>0.022130</td>
    </tr>
    <tr>
      <th>2020-12-24</th>
      <td>74100</td>
      <td>78800</td>
      <td>74000</td>
      <td>77800</td>
      <td>32317535</td>
      <td>0.052774</td>
    </tr>
  </tbody>
</table>
</div>



**Apple**도 가져올 수 있습니다.


```python
# Apple(AAPL), 애플
apple = fdr.DataReader('AAPL')
```


```python
apple.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-12-18</th>
      <td>126.65</td>
      <td>128.96</td>
      <td>129.10</td>
      <td>126.12</td>
      <td>192540000.0</td>
      <td>-0.0159</td>
    </tr>
    <tr>
      <th>2020-12-21</th>
      <td>128.23</td>
      <td>125.03</td>
      <td>128.26</td>
      <td>123.47</td>
      <td>121250000.0</td>
      <td>0.0124</td>
    </tr>
    <tr>
      <th>2020-12-22</th>
      <td>131.88</td>
      <td>131.68</td>
      <td>134.40</td>
      <td>129.66</td>
      <td>169350000.0</td>
      <td>0.0285</td>
    </tr>
    <tr>
      <th>2020-12-23</th>
      <td>130.96</td>
      <td>132.18</td>
      <td>132.32</td>
      <td>130.83</td>
      <td>88220000.0</td>
      <td>-0.0070</td>
    </tr>
    <tr>
      <th>2020-12-24</th>
      <td>131.99</td>
      <td>131.19</td>
      <td>133.46</td>
      <td>131.10</td>
      <td>52790000.0</td>
      <td>0.0079</td>
    </tr>
  </tbody>
</table>
</div>



다음과 같이 `2017`을 같이 넘겨주면, 해당 시점 이후의 주식 데이터를 가져옵니다.


```python
# Apple(AAPL), 애플
apple = fdr.DataReader('AAPL', '2017')
```


```python
apple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-03</th>
      <td>29.04</td>
      <td>28.95</td>
      <td>29.08</td>
      <td>28.69</td>
      <td>115130000.0</td>
      <td>0.0031</td>
    </tr>
    <tr>
      <th>2017-01-04</th>
      <td>29.00</td>
      <td>28.96</td>
      <td>29.13</td>
      <td>28.94</td>
      <td>84470000.0</td>
      <td>-0.0014</td>
    </tr>
    <tr>
      <th>2017-01-05</th>
      <td>29.15</td>
      <td>28.98</td>
      <td>29.22</td>
      <td>28.95</td>
      <td>88770000.0</td>
      <td>0.0052</td>
    </tr>
    <tr>
      <th>2017-01-06</th>
      <td>29.48</td>
      <td>29.20</td>
      <td>29.54</td>
      <td>29.12</td>
      <td>127010000.0</td>
      <td>0.0113</td>
    </tr>
    <tr>
      <th>2017-01-09</th>
      <td>29.75</td>
      <td>29.49</td>
      <td>29.86</td>
      <td>29.48</td>
      <td>134250000.0</td>
      <td>0.0092</td>
    </tr>
  </tbody>
</table>
</div>



**시작**과 **끝** 날짜를 지정하여 범위 데이터를 가져올 수 있습니다.


```python
# Ford(F), 1980-01-01 ~ 2019-12-30 (40년 데이터)
ford = fdr.DataReader('F', '1980-01-01', '2019-12-30')
```


```python
# 삼성전자 주식코드: 005930
STOCK_CODE = '005930'
```


```python
stock = fdr.DataReader(STOCK_CODE)
```


```python
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-01-20</th>
      <td>800</td>
      <td>844</td>
      <td>800</td>
      <td>838</td>
      <td>91310</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1997-01-21</th>
      <td>844</td>
      <td>844</td>
      <td>803</td>
      <td>809</td>
      <td>81800</td>
      <td>-0.034606</td>
    </tr>
    <tr>
      <th>1997-01-22</th>
      <td>805</td>
      <td>805</td>
      <td>782</td>
      <td>786</td>
      <td>81910</td>
      <td>-0.028430</td>
    </tr>
    <tr>
      <th>1997-01-23</th>
      <td>786</td>
      <td>798</td>
      <td>770</td>
      <td>776</td>
      <td>74200</td>
      <td>-0.012723</td>
    </tr>
    <tr>
      <th>1997-01-24</th>
      <td>745</td>
      <td>793</td>
      <td>745</td>
      <td>783</td>
      <td>98260</td>
      <td>0.009021</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-12-18</th>
      <td>73300</td>
      <td>73700</td>
      <td>73000</td>
      <td>73000</td>
      <td>17613029</td>
      <td>-0.004093</td>
    </tr>
    <tr>
      <th>2020-12-21</th>
      <td>73100</td>
      <td>73400</td>
      <td>72000</td>
      <td>73000</td>
      <td>20367355</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-12-22</th>
      <td>72500</td>
      <td>73200</td>
      <td>72100</td>
      <td>72300</td>
      <td>16304910</td>
      <td>-0.009589</td>
    </tr>
    <tr>
      <th>2020-12-23</th>
      <td>72400</td>
      <td>74000</td>
      <td>72300</td>
      <td>73900</td>
      <td>19411326</td>
      <td>0.022130</td>
    </tr>
    <tr>
      <th>2020-12-24</th>
      <td>74100</td>
      <td>78800</td>
      <td>74000</td>
      <td>77800</td>
      <td>32317535</td>
      <td>0.052774</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock.index
```




    DatetimeIndex(['1997-01-20', '1997-01-21', '1997-01-22', '1997-01-23',
                   '1997-01-24', '1997-01-25', '1997-01-27', '1997-01-28',
                   '1997-01-29', '1997-01-30',
                   ...
                   '2020-12-11', '2020-12-14', '2020-12-15', '2020-12-16',
                   '2020-12-17', '2020-12-18', '2020-12-21', '2020-12-22',
                   '2020-12-23', '2020-12-24'],
                  dtype='datetime64[ns]', name='Date', length=6000, freq=None)



위에서 보시는 바와 같이 index가 `DatetimeIndex`로 지정되어 있습니다.

`DatetimeIndex`로 정의되어 있다면, 아래와 같이 연도, 월, 일을 쪼갤 수 있으며, **월별, 연도별 피벗데이터**를 만들때 유용하게 활용할 수 있습니다.


```python
stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day
```


```python
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-01-20</th>
      <td>800</td>
      <td>844</td>
      <td>800</td>
      <td>838</td>
      <td>91310</td>
      <td>NaN</td>
      <td>1997</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1997-01-21</th>
      <td>844</td>
      <td>844</td>
      <td>803</td>
      <td>809</td>
      <td>81800</td>
      <td>-0.034606</td>
      <td>1997</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1997-01-22</th>
      <td>805</td>
      <td>805</td>
      <td>782</td>
      <td>786</td>
      <td>81910</td>
      <td>-0.028430</td>
      <td>1997</td>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1997-01-23</th>
      <td>786</td>
      <td>798</td>
      <td>770</td>
      <td>776</td>
      <td>74200</td>
      <td>-0.012723</td>
      <td>1997</td>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1997-01-24</th>
      <td>745</td>
      <td>793</td>
      <td>745</td>
      <td>783</td>
      <td>98260</td>
      <td>0.009021</td>
      <td>1997</td>
      <td>1</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



## 시각화


```python
plt.figure(figsize=(16, 9))
sns.lineplot(y=stock['Close'], x=stock.index)
plt.xlabel('time')
plt.ylabel('price')
```




    Text(0, 0.5, 'price')




![png](output_24_1.png)



```python
time_steps = [['1990', '2000'], 
              ['2000', '2010'], 
              ['2010', '2015'], 
              ['2015', '2020']]

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 9)
for i in range(4):
    ax = axes[i//2, i%2]
    df = stock.loc[(stock.index > time_steps[i][0]) & (stock.index < time_steps[i][1])]
    sns.lineplot(y=df['Close'], x=df.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')
plt.tight_layout()
plt.show()
```


![png](output_25_0.png)


## 데이터 전처리


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# 스케일 후 columns
scaled = scaler.fit_transform(stock[scale_cols])
scaled
```




    array([[0.01079622, 0.01071066, 0.01081081, 0.00273412, 0.00143815],
           [0.01139001, 0.01071066, 0.01085135, 0.00235834, 0.00128837],
           [0.0108637 , 0.01021574, 0.01056757, 0.00206031, 0.0012901 ],
           ...,
           [0.97840756, 0.92893401, 0.97432432, 0.92873155, 0.25680619],
           [0.97705803, 0.93908629, 0.97702703, 0.94946419, 0.30573298],
           [1.        , 1.        , 1.        , 1.        , 0.50900883]])




```python
df = pd.DataFrame(scaled, columns=scale_cols)
```

## train / test 분할


```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)
```


```python
x_train.shape, y_train.shape
```




    ((4800, 4), (4800,))




```python
x_test.shape, y_test.shape
```




    ((1200, 4), (1200,))




```python
x_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.010796</td>
      <td>0.010711</td>
      <td>0.010811</td>
      <td>0.001438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.011390</td>
      <td>0.010711</td>
      <td>0.010851</td>
      <td>0.001288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.010864</td>
      <td>0.010216</td>
      <td>0.010568</td>
      <td>0.001290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.010607</td>
      <td>0.010127</td>
      <td>0.010405</td>
      <td>0.001169</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.010054</td>
      <td>0.010063</td>
      <td>0.010068</td>
      <td>0.001548</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4795</th>
      <td>0.307692</td>
      <td>0.291878</td>
      <td>0.301622</td>
      <td>0.006883</td>
    </tr>
    <tr>
      <th>4796</th>
      <td>0.310931</td>
      <td>0.295178</td>
      <td>0.311081</td>
      <td>0.004095</td>
    </tr>
    <tr>
      <th>4797</th>
      <td>0.313360</td>
      <td>0.295939</td>
      <td>0.310000</td>
      <td>0.002620</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>0.310391</td>
      <td>0.292386</td>
      <td>0.307297</td>
      <td>0.002749</td>
    </tr>
    <tr>
      <th>4799</th>
      <td>0.310391</td>
      <td>0.294670</td>
      <td>0.310270</td>
      <td>0.003905</td>
    </tr>
  </tbody>
</table>
<p>4800 rows × 4 columns</p>
</div>



## TensroFlow Dataset을 활용한 시퀀스 데이터셋 구성


```python
import tensorflow as tf
```


```python
def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)
```

Hyperparameter를 정의합니다.


```python
WINDOW_SIZE=20
BATCH_SIZE=32
```


```python
# trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
```


```python
# 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있습니다.
# X: (batch_size, window_size, feature)
# Y: (batch_size, feature)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')
```

    데이터셋(X) 구성(batch_size, window_size, feature갯수): (32, 20, 1)
    데이터셋(Y) 구성(batch_size, window_size, feature갯수): (32, 1)
    

## 모델


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])
```


```python
# Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])
```


```python
# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)
```


```python
history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])
```

    Epoch 1/50
        145/Unknown - 1s 6ms/step - loss: 1.3915e-04 - mse: 2.7831e-04
    Epoch 00001: val_loss improved from inf to 0.00442, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 2s 11ms/step - loss: 1.3702e-04 - mse: 2.7404e-04 - val_loss: 0.0044 - val_mse: 0.0088
    Epoch 2/50
    150/150 [==============================] - ETA: 0s - loss: 3.5403e-05 - mse: 7.0805e-05
    Epoch 00002: val_loss improved from 0.00442 to 0.00373, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 3.5403e-05 - mse: 7.0805e-05 - val_loss: 0.0037 - val_mse: 0.0075
    Epoch 3/50
    145/150 [============================>.] - ETA: 0s - loss: 3.0890e-05 - mse: 6.1779e-05
    Epoch 00003: val_loss improved from 0.00373 to 0.00265, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 3.2327e-05 - mse: 6.4655e-05 - val_loss: 0.0026 - val_mse: 0.0053
    Epoch 4/50
    139/150 [==========================>...] - ETA: 0s - loss: 3.0849e-05 - mse: 6.1697e-05
    Epoch 00004: val_loss did not improve from 0.00265
    150/150 [==============================] - 1s 8ms/step - loss: 3.3384e-05 - mse: 6.6768e-05 - val_loss: 0.0028 - val_mse: 0.0056
    Epoch 5/50
    140/150 [===========================>..] - ETA: 0s - loss: 2.9329e-05 - mse: 5.8658e-05
    Epoch 00005: val_loss improved from 0.00265 to 0.00255, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 3.1223e-05 - mse: 6.2445e-05 - val_loss: 0.0026 - val_mse: 0.0051
    Epoch 6/50
    139/150 [==========================>...] - ETA: 0s - loss: 2.9247e-05 - mse: 5.8493e-05
    Epoch 00006: val_loss improved from 0.00255 to 0.00206, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 3.1141e-05 - mse: 6.2282e-05 - val_loss: 0.0021 - val_mse: 0.0041
    Epoch 7/50
    146/150 [============================>.] - ETA: 0s - loss: 2.9408e-05 - mse: 5.8817e-05
    Epoch 00007: val_loss did not improve from 0.00206
    150/150 [==============================] - 1s 8ms/step - loss: 2.9931e-05 - mse: 5.9862e-05 - val_loss: 0.0022 - val_mse: 0.0044
    Epoch 8/50
    139/150 [==========================>...] - ETA: 0s - loss: 2.6324e-05 - mse: 5.2648e-05
    Epoch 00008: val_loss did not improve from 0.00206
    150/150 [==============================] - 1s 8ms/step - loss: 2.8588e-05 - mse: 5.7175e-05 - val_loss: 0.0022 - val_mse: 0.0043
    Epoch 9/50
    147/150 [============================>.] - ETA: 0s - loss: 2.6072e-05 - mse: 5.2143e-05
    Epoch 00009: val_loss improved from 0.00206 to 0.00140, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 2.6393e-05 - mse: 5.2786e-05 - val_loss: 0.0014 - val_mse: 0.0028
    Epoch 10/50
    142/150 [===========================>..] - ETA: 0s - loss: 2.5399e-05 - mse: 5.0798e-05
    Epoch 00010: val_loss did not improve from 0.00140
    150/150 [==============================] - 1s 8ms/step - loss: 2.6429e-05 - mse: 5.2859e-05 - val_loss: 0.0015 - val_mse: 0.0031
    Epoch 11/50
    146/150 [============================>.] - ETA: 0s - loss: 2.4973e-05 - mse: 4.9946e-05
    Epoch 00011: val_loss did not improve from 0.00140
    150/150 [==============================] - 1s 8ms/step - loss: 2.5656e-05 - mse: 5.1313e-05 - val_loss: 0.0019 - val_mse: 0.0038
    Epoch 12/50
    143/150 [===========================>..] - ETA: 0s - loss: 2.3122e-05 - mse: 4.6245e-05
    Epoch 00012: val_loss improved from 0.00140 to 0.00131, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 2.4026e-05 - mse: 4.8053e-05 - val_loss: 0.0013 - val_mse: 0.0026
    Epoch 13/50
    145/150 [============================>.] - ETA: 0s - loss: 2.3306e-05 - mse: 4.6611e-05
    Epoch 00013: val_loss improved from 0.00131 to 0.00073, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 2.3637e-05 - mse: 4.7273e-05 - val_loss: 7.2618e-04 - val_mse: 0.0015
    Epoch 14/50
    146/150 [============================>.] - ETA: 0s - loss: 2.3101e-05 - mse: 4.6202e-05
    Epoch 00014: val_loss did not improve from 0.00073
    150/150 [==============================] - 1s 8ms/step - loss: 2.3758e-05 - mse: 4.7515e-05 - val_loss: 0.0011 - val_mse: 0.0021
    Epoch 15/50
    138/150 [==========================>...] - ETA: 0s - loss: 2.0612e-05 - mse: 4.1225e-05
    Epoch 00015: val_loss did not improve from 0.00073
    150/150 [==============================] - 1s 8ms/step - loss: 2.2132e-05 - mse: 4.4265e-05 - val_loss: 9.7743e-04 - val_mse: 0.0020
    Epoch 16/50
    145/150 [============================>.] - ETA: 0s - loss: 2.1968e-05 - mse: 4.3937e-05
    Epoch 00016: val_loss improved from 0.00073 to 0.00068, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 2.2618e-05 - mse: 4.5237e-05 - val_loss: 6.7990e-04 - val_mse: 0.0014
    Epoch 17/50
    140/150 [===========================>..] - ETA: 0s - loss: 2.0500e-05 - mse: 4.1001e-05
    Epoch 00017: val_loss improved from 0.00068 to 0.00051, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 2.2256e-05 - mse: 4.4512e-05 - val_loss: 5.1027e-04 - val_mse: 0.0010
    Epoch 18/50
    141/150 [===========================>..] - ETA: 0s - loss: 1.9266e-05 - mse: 3.8533e-05
    Epoch 00018: val_loss improved from 0.00051 to 0.00047, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 2.0363e-05 - mse: 4.0727e-05 - val_loss: 4.7082e-04 - val_mse: 9.4163e-04
    Epoch 19/50
    140/150 [===========================>..] - ETA: 0s - loss: 1.8217e-05 - mse: 3.6433e-05
    Epoch 00019: val_loss improved from 0.00047 to 0.00039, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.8908e-05 - mse: 3.7815e-05 - val_loss: 3.9196e-04 - val_mse: 7.8393e-04
    Epoch 20/50
    146/150 [============================>.] - ETA: 0s - loss: 1.7757e-05 - mse: 3.5514e-05
    Epoch 00020: val_loss did not improve from 0.00039
    150/150 [==============================] - 1s 8ms/step - loss: 1.8188e-05 - mse: 3.6375e-05 - val_loss: 5.7883e-04 - val_mse: 0.0012
    Epoch 21/50
    143/150 [===========================>..] - ETA: 0s - loss: 1.8277e-05 - mse: 3.6553e-05
    Epoch 00021: val_loss improved from 0.00039 to 0.00037, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 9ms/step - loss: 1.9064e-05 - mse: 3.8128e-05 - val_loss: 3.6557e-04 - val_mse: 7.3113e-04
    Epoch 22/50
    149/150 [============================>.] - ETA: 0s - loss: 1.7973e-05 - mse: 3.5946e-05
    Epoch 00022: val_loss did not improve from 0.00037
    150/150 [==============================] - 1s 8ms/step - loss: 1.7972e-05 - mse: 3.5943e-05 - val_loss: 7.2105e-04 - val_mse: 0.0014
    Epoch 23/50
    144/150 [===========================>..] - ETA: 0s - loss: 1.6506e-05 - mse: 3.3012e-05
    Epoch 00023: val_loss improved from 0.00037 to 0.00022, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.6973e-05 - mse: 3.3946e-05 - val_loss: 2.2470e-04 - val_mse: 4.4941e-04
    Epoch 24/50
    143/150 [===========================>..] - ETA: 0s - loss: 1.5503e-05 - mse: 3.1005e-05
    Epoch 00024: val_loss improved from 0.00022 to 0.00020, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.6053e-05 - mse: 3.2106e-05 - val_loss: 1.9826e-04 - val_mse: 3.9652e-04
    Epoch 25/50
    141/150 [===========================>..] - ETA: 0s - loss: 1.4575e-05 - mse: 2.9149e-05
    Epoch 00025: val_loss did not improve from 0.00020
    150/150 [==============================] - 1s 8ms/step - loss: 1.5453e-05 - mse: 3.0907e-05 - val_loss: 3.3893e-04 - val_mse: 6.7787e-04
    Epoch 26/50
    141/150 [===========================>..] - ETA: 0s - loss: 1.4477e-05 - mse: 2.8954e-05
    Epoch 00026: val_loss did not improve from 0.00020
    150/150 [==============================] - 1s 8ms/step - loss: 1.5228e-05 - mse: 3.0457e-05 - val_loss: 3.3818e-04 - val_mse: 6.7637e-04
    Epoch 27/50
    146/150 [============================>.] - ETA: 0s - loss: 1.5209e-05 - mse: 3.0417e-05
    Epoch 00027: val_loss did not improve from 0.00020
    150/150 [==============================] - 1s 8ms/step - loss: 1.5293e-05 - mse: 3.0586e-05 - val_loss: 2.2337e-04 - val_mse: 4.4673e-04
    Epoch 28/50
    141/150 [===========================>..] - ETA: 0s - loss: 1.4172e-05 - mse: 2.8343e-05
    Epoch 00028: val_loss improved from 0.00020 to 0.00017, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.5115e-05 - mse: 3.0230e-05 - val_loss: 1.6553e-04 - val_mse: 3.3106e-04
    Epoch 29/50
    145/150 [============================>.] - ETA: 0s - loss: 1.4089e-05 - mse: 2.8178e-05
    Epoch 00029: val_loss improved from 0.00017 to 0.00016, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.4461e-05 - mse: 2.8921e-05 - val_loss: 1.5889e-04 - val_mse: 3.1778e-04
    Epoch 30/50
    144/150 [===========================>..] - ETA: 0s - loss: 1.5103e-05 - mse: 3.0207e-05
    Epoch 00030: val_loss improved from 0.00016 to 0.00016, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.5549e-05 - mse: 3.1097e-05 - val_loss: 1.5603e-04 - val_mse: 3.1205e-04
    Epoch 31/50
    139/150 [==========================>...] - ETA: 0s - loss: 1.4528e-05 - mse: 2.9055e-05
    Epoch 00031: val_loss did not improve from 0.00016
    150/150 [==============================] - 1s 8ms/step - loss: 1.6304e-05 - mse: 3.2608e-05 - val_loss: 1.8720e-04 - val_mse: 3.7439e-04
    Epoch 32/50
    142/150 [===========================>..] - ETA: 0s - loss: 1.4269e-05 - mse: 2.8537e-05
    Epoch 00032: val_loss improved from 0.00016 to 0.00015, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 9ms/step - loss: 1.4625e-05 - mse: 2.9250e-05 - val_loss: 1.4684e-04 - val_mse: 2.9369e-04
    Epoch 33/50
    142/150 [===========================>..] - ETA: 0s - loss: 1.2597e-05 - mse: 2.5194e-05
    Epoch 00033: val_loss did not improve from 0.00015
    150/150 [==============================] - 1s 8ms/step - loss: 1.3262e-05 - mse: 2.6524e-05 - val_loss: 1.8885e-04 - val_mse: 3.7770e-04
    Epoch 34/50
    146/150 [============================>.] - ETA: 0s - loss: 1.4380e-05 - mse: 2.8761e-05
    Epoch 00034: val_loss improved from 0.00015 to 0.00014, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.4529e-05 - mse: 2.9058e-05 - val_loss: 1.4084e-04 - val_mse: 2.8168e-04
    Epoch 35/50
    144/150 [===========================>..] - ETA: 0s - loss: 1.2255e-05 - mse: 2.4510e-05
    Epoch 00035: val_loss improved from 0.00014 to 0.00014, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.2630e-05 - mse: 2.5259e-05 - val_loss: 1.3570e-04 - val_mse: 2.7141e-04
    Epoch 36/50
    146/150 [============================>.] - ETA: 0s - loss: 1.2402e-05 - mse: 2.4805e-05
    Epoch 00036: val_loss did not improve from 0.00014
    150/150 [==============================] - 1s 8ms/step - loss: 1.2712e-05 - mse: 2.5423e-05 - val_loss: 1.3717e-04 - val_mse: 2.7434e-04
    Epoch 37/50
    148/150 [============================>.] - ETA: 0s - loss: 1.2014e-05 - mse: 2.4028e-05
    Epoch 00037: val_loss improved from 0.00014 to 0.00013, saving model to tmp/ckeckpointer.ckpt
    150/150 [==============================] - 1s 8ms/step - loss: 1.2155e-05 - mse: 2.4311e-05 - val_loss: 1.3075e-04 - val_mse: 2.6150e-04
    Epoch 38/50
    149/150 [============================>.] - ETA: 0s - loss: 1.2660e-05 - mse: 2.5319e-05
    Epoch 00038: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.2697e-05 - mse: 2.5395e-05 - val_loss: 3.2183e-04 - val_mse: 6.4366e-04
    Epoch 39/50
    139/150 [==========================>...] - ETA: 0s - loss: 1.0822e-05 - mse: 2.1644e-05
    Epoch 00039: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.1924e-05 - mse: 2.3849e-05 - val_loss: 3.5369e-04 - val_mse: 7.0737e-04
    Epoch 40/50
    143/150 [===========================>..] - ETA: 0s - loss: 1.1230e-05 - mse: 2.2460e-05
    Epoch 00040: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.1633e-05 - mse: 2.3266e-05 - val_loss: 2.4469e-04 - val_mse: 4.8938e-04
    Epoch 41/50
    140/150 [===========================>..] - ETA: 0s - loss: 1.2174e-05 - mse: 2.4348e-05
    Epoch 00041: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.2546e-05 - mse: 2.5092e-05 - val_loss: 2.2481e-04 - val_mse: 4.4961e-04
    Epoch 42/50
    141/150 [===========================>..] - ETA: 0s - loss: 1.0771e-05 - mse: 2.1541e-05
    Epoch 00042: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.1201e-05 - mse: 2.2402e-05 - val_loss: 1.9641e-04 - val_mse: 3.9283e-04
    Epoch 43/50
    150/150 [==============================] - ETA: 0s - loss: 1.1375e-05 - mse: 2.2750e-05
    Epoch 00043: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.1375e-05 - mse: 2.2750e-05 - val_loss: 5.3345e-04 - val_mse: 0.0011
    Epoch 44/50
    147/150 [============================>.] - ETA: 0s - loss: 1.0875e-05 - mse: 2.1749e-05
    Epoch 00044: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.0975e-05 - mse: 2.1949e-05 - val_loss: 3.9328e-04 - val_mse: 7.8657e-04
    Epoch 45/50
    144/150 [===========================>..] - ETA: 0s - loss: 1.0671e-05 - mse: 2.1343e-05
    Epoch 00045: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.1074e-05 - mse: 2.2147e-05 - val_loss: 3.3973e-04 - val_mse: 6.7945e-04
    Epoch 46/50
    143/150 [===========================>..] - ETA: 0s - loss: 1.0369e-05 - mse: 2.0738e-05
    Epoch 00046: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.0733e-05 - mse: 2.1465e-05 - val_loss: 3.1263e-04 - val_mse: 6.2526e-04
    Epoch 47/50
    149/150 [============================>.] - ETA: 0s - loss: 1.1758e-05 - mse: 2.3515e-05
    Epoch 00047: val_loss did not improve from 0.00013
    150/150 [==============================] - 1s 8ms/step - loss: 1.1775e-05 - mse: 2.3549e-05 - val_loss: 1.3305e-04 - val_mse: 2.6610e-04
    

저장한 ModelCheckpoint 를 로드합니다.


```python
model.load_weights(filename)
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f3cdd729518>



`test_data`를 활용하여 예측을 진행합니다.


```python
pred = model.predict(test_data)
```


```python
pred.shape
```




    (1180, 1)



## 예측 데이터 시각화


```python
plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
```


![png](output_53_0.png)



```python

```

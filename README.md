# 2024 날씨 빅데이터 콘테스트 

## 전력 분야 - 기상에 따른 공동주택 전력수요 예측 개선 


```python
import pandas as pd
import os 
```


```python
df = pd.read_csv('./data/electric_train_cp949.csv', encoding='cp949', index_col=0)
df['electric_train.tm'] = pd.to_datetime(df['electric_train.tm'])

print(df.shape)
```

    (7593355, 16)
    


```python
df.head()
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
      <th>electric_train.num</th>
      <th>electric_train.tm</th>
      <th>electric_train.hh24</th>
      <th>electric_train.n</th>
      <th>electric_train.stn</th>
      <th>electric_train.sum_qctr</th>
      <th>electric_train.sum_load</th>
      <th>electric_train.n_mean_load</th>
      <th>electric_train.nph_ta</th>
      <th>electric_train.nph_hm</th>
      <th>electric_train.nph_ws_10m</th>
      <th>electric_train.nph_rn_60m</th>
      <th>electric_train.nph_ta_chi</th>
      <th>electric_train.weekday</th>
      <th>electric_train.week_name</th>
      <th>electric_train.elec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4821</td>
      <td>2021-01-01 01:00:00</td>
      <td>1</td>
      <td>11</td>
      <td>884</td>
      <td>6950</td>
      <td>751.32</td>
      <td>68.606449</td>
      <td>2.2</td>
      <td>62.7</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>4</td>
      <td>0</td>
      <td>99.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4821</td>
      <td>2021-01-01 02:00:00</td>
      <td>2</td>
      <td>11</td>
      <td>884</td>
      <td>6950</td>
      <td>692.60</td>
      <td>68.606449</td>
      <td>2.3</td>
      <td>63.1</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>-0.6</td>
      <td>4</td>
      <td>0</td>
      <td>91.78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4821</td>
      <td>2021-01-01 03:00:00</td>
      <td>3</td>
      <td>11</td>
      <td>884</td>
      <td>6950</td>
      <td>597.48</td>
      <td>68.606449</td>
      <td>2.2</td>
      <td>62.4</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>-1.3</td>
      <td>4</td>
      <td>0</td>
      <td>79.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4821</td>
      <td>2021-01-01 04:00:00</td>
      <td>4</td>
      <td>11</td>
      <td>884</td>
      <td>6950</td>
      <td>553.48</td>
      <td>68.606449</td>
      <td>1.7</td>
      <td>63.5</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>-0.2</td>
      <td>4</td>
      <td>0</td>
      <td>73.34</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4821</td>
      <td>2021-01-01 05:00:00</td>
      <td>5</td>
      <td>11</td>
      <td>884</td>
      <td>6950</td>
      <td>526.24</td>
      <td>68.606449</td>
      <td>1.7</td>
      <td>63.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>-0.8</td>
      <td>4</td>
      <td>0</td>
      <td>69.73</td>
    </tr>
  </tbody>
</table>
</div>



전력 데이터 

![image](https://github.com/khw11044/csv_dataset/assets/51473705/83b214a1-8661-465b-b176-6912d867f856)


```python
elec_cols = ['electric_train.'+ a for a in ['tm', 'hh24', 'weekday', 'week_name', 'sum_qctr', 'n', 'sum_load', 'n_mean_load', 'elec']]

elec_cols
```




    ['electric_train.tm',
     'electric_train.hh24',
     'electric_train.weekday',
     'electric_train.week_name',
     'electric_train.sum_qctr',
     'electric_train.n',
     'electric_train.sum_load',
     'electric_train.n_mean_load',
     'electric_train.elec']




```python
df_elec = df[elec_cols]
df_elec.head()
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
      <th>electric_train.tm</th>
      <th>electric_train.hh24</th>
      <th>electric_train.weekday</th>
      <th>electric_train.week_name</th>
      <th>electric_train.sum_qctr</th>
      <th>electric_train.n</th>
      <th>electric_train.sum_load</th>
      <th>electric_train.n_mean_load</th>
      <th>electric_train.elec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-01-01 01:00:00</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>751.32</td>
      <td>68.606449</td>
      <td>99.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-01 02:00:00</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>692.60</td>
      <td>68.606449</td>
      <td>91.78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-01 03:00:00</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>597.48</td>
      <td>68.606449</td>
      <td>79.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-01 04:00:00</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>553.48</td>
      <td>68.606449</td>
      <td>73.34</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-01-01 05:00:00</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>526.24</td>
      <td>68.606449</td>
      <td>69.73</td>
    </tr>
  </tbody>
</table>
</div>



기상 데이터 

![image](https://github.com/khw11044/csv_dataset/assets/51473705/cc9f5973-b068-4c77-812a-dc68160cd285)


```python
weat_cols = ['electric_train.'+ a for a in ['num', 'stn', 'nph_ta', 'nph_hm', 'nph_ws_10m', 'nph_rn_60m', 'nph_ta_chi']]

weat_cols 
```




    ['electric_train.num',
     'electric_train.stn',
     'electric_train.nph_ta',
     'electric_train.nph_hm',
     'electric_train.nph_ws_10m',
     'electric_train.nph_rn_60m',
     'electric_train.nph_ta_chi']




```python
df_weat = df[weat_cols]
df_weat.head()
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
      <th>electric_train.num</th>
      <th>electric_train.stn</th>
      <th>electric_train.nph_ta</th>
      <th>electric_train.nph_hm</th>
      <th>electric_train.nph_ws_10m</th>
      <th>electric_train.nph_rn_60m</th>
      <th>electric_train.nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4821</td>
      <td>884</td>
      <td>2.2</td>
      <td>62.7</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4821</td>
      <td>884</td>
      <td>2.3</td>
      <td>63.1</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>-0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4821</td>
      <td>884</td>
      <td>2.2</td>
      <td>62.4</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>-1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4821</td>
      <td>884</td>
      <td>1.7</td>
      <td>63.5</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>-0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4821</td>
      <td>884</td>
      <td>1.7</td>
      <td>63.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>-0.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.concat([df_elec, df_weat], axis=1)

df.head()
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
      <th>electric_train.tm</th>
      <th>electric_train.hh24</th>
      <th>electric_train.weekday</th>
      <th>electric_train.week_name</th>
      <th>electric_train.sum_qctr</th>
      <th>electric_train.n</th>
      <th>electric_train.sum_load</th>
      <th>electric_train.n_mean_load</th>
      <th>electric_train.elec</th>
      <th>electric_train.num</th>
      <th>electric_train.stn</th>
      <th>electric_train.nph_ta</th>
      <th>electric_train.nph_hm</th>
      <th>electric_train.nph_ws_10m</th>
      <th>electric_train.nph_rn_60m</th>
      <th>electric_train.nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-01-01 01:00:00</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>751.32</td>
      <td>68.606449</td>
      <td>99.56</td>
      <td>4821</td>
      <td>884</td>
      <td>2.2</td>
      <td>62.7</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-01 02:00:00</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>692.60</td>
      <td>68.606449</td>
      <td>91.78</td>
      <td>4821</td>
      <td>884</td>
      <td>2.3</td>
      <td>63.1</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>-0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-01 03:00:00</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>597.48</td>
      <td>68.606449</td>
      <td>79.17</td>
      <td>4821</td>
      <td>884</td>
      <td>2.2</td>
      <td>62.4</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>-1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-01 04:00:00</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>553.48</td>
      <td>68.606449</td>
      <td>73.34</td>
      <td>4821</td>
      <td>884</td>
      <td>1.7</td>
      <td>63.5</td>
      <td>1.7</td>
      <td>0.0</td>
      <td>-0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-01-01 05:00:00</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>6950</td>
      <td>11</td>
      <td>526.24</td>
      <td>68.606449</td>
      <td>69.73</td>
      <td>4821</td>
      <td>884</td>
      <td>1.7</td>
      <td>63.0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>-0.8</td>
    </tr>
  </tbody>
</table>
</div>



![image](https://github.com/khw11044/csv_dataset/assets/51473705/c212e653-dcdc-49b3-bf75-ea356c7db0df)

## **전력기상지수** 

기상변화에 따른 지역별 공동주택의 예상되는 전력수요 변화를 기상예보처럼 국민들이 쉽게 인지할 수 있도록 수치화하여 최대 72h 예측해주는 서비스 

### **산출식**
해당지역 (5km x 5km) 공동주택의 연중 평균부하를 100으로 하였을 때, 특정시각의 전력수요 또는 예상전력수요를 상대비율로 표현한 값 

$\text{A격자 00시간의 전력기상지수} = \frac{A격자 00시각의 전력수요 (또는 예상 전력수요)}{A격자 해당년도 전시간 평균 전력수요}$

$\text{elec}_{i} = \frac{\text{sum load}_{i}}{2}$

$\text{elec}_{i} = \text{sum load}_{i} \text{n mean load}_{i-1}$

### **활용법**

전일 또는 지난주 대비 지수의 증감비율을 통해 전력수요 변화량 예상 

**예시)** 
- 전일 최고 전력기상지수가 100, 
- 당일 최고 전력기상지수가 125,
- 당일 최대수요는 전일대비 125/100 = 1.25배 (25%) 증가를 예상해 활용 


$\text{당일 최대수요 증감소율}_{i} = \frac{\text{당일 최고 전력기상지수}(\text{max}(elec_{i}))}{\text{전일 최고 전력기상지수}(\text{max}(elec_{i-1}))}$

- elec: 전력기상지수 
- sum_qctr: 계약전력합계
- n: 공동주택 수
- sum_load: 전력수요 합계 
- n_mean_load: 전력부하량 평균 

### **검증 데이터**
**검증 데이터**는 전력기상지수를 산출할 수 있는 변수인 sum_qctr(계약전력합계), n(공동주택 수), sum_load(전력수요 합계), n_mean_load(전력부하량 평균)를 제외하고 제공 

$\text{elec}_{i} = \frac{\text{(sum load)}_{i}}{\text{(n mean load)}_{i-1}}$


## **목표**
공동주택 전력수요 예측 

1. 공공데이터를 활용하여 공동주택 전력수요 증감 영향 요인 분석 
2. 계절, 지역에 따른 모델 세분화를 통한 공동주택 전력 수요 예측 (전력기상지수(electric_train.elec)) 최적모델 개발 


```python
df_train = df[df['electric_train.tm'] < '2023-01-01']
df_val = df[df['electric_train.tm'] >= '2023-01-01'].drop(['electric_train.sum_qctr', 'electric_train.n', 'electric_train.sum_load', 'electric_train.n_mean_load'], axis=1)
# df_val는 다른 features에서 계산을 통해 구한 후 채워넣야 할듯? <-- 훈련 데이터에서는 사용하기 때문 (내 생각.....)
```


```python
# 훈련데이터
df_train.tail()
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
      <th>electric_train.tm</th>
      <th>electric_train.hh24</th>
      <th>electric_train.weekday</th>
      <th>electric_train.week_name</th>
      <th>electric_train.sum_qctr</th>
      <th>electric_train.n</th>
      <th>electric_train.sum_load</th>
      <th>electric_train.n_mean_load</th>
      <th>electric_train.elec</th>
      <th>electric_train.num</th>
      <th>electric_train.stn</th>
      <th>electric_train.nph_ta</th>
      <th>electric_train.nph_hm</th>
      <th>electric_train.nph_ws_10m</th>
      <th>electric_train.nph_rn_60m</th>
      <th>electric_train.nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7593350</th>
      <td>2022-12-31 19:00:00</td>
      <td>19</td>
      <td>5</td>
      <td>1</td>
      <td>34200</td>
      <td>23</td>
      <td>6851.72</td>
      <td>225.461986</td>
      <td>132.13</td>
      <td>20947</td>
      <td>671</td>
      <td>2.7</td>
      <td>45.4</td>
      <td>3.2</td>
      <td>0.0</td>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>7593351</th>
      <td>2022-12-31 20:00:00</td>
      <td>20</td>
      <td>5</td>
      <td>1</td>
      <td>34200</td>
      <td>23</td>
      <td>6779.84</td>
      <td>225.461986</td>
      <td>130.74</td>
      <td>20947</td>
      <td>671</td>
      <td>2.7</td>
      <td>46.3</td>
      <td>3.1</td>
      <td>0.0</td>
      <td>-0.4</td>
    </tr>
    <tr>
      <th>7593352</th>
      <td>2022-12-31 21:00:00</td>
      <td>21</td>
      <td>5</td>
      <td>1</td>
      <td>34200</td>
      <td>23</td>
      <td>6802.40</td>
      <td>225.461986</td>
      <td>131.18</td>
      <td>20947</td>
      <td>671</td>
      <td>2.6</td>
      <td>46.8</td>
      <td>3.1</td>
      <td>0.0</td>
      <td>-0.5</td>
    </tr>
    <tr>
      <th>7593353</th>
      <td>2022-12-31 22:00:00</td>
      <td>22</td>
      <td>5</td>
      <td>1</td>
      <td>34200</td>
      <td>23</td>
      <td>6706.68</td>
      <td>225.461986</td>
      <td>129.33</td>
      <td>20947</td>
      <td>671</td>
      <td>2.4</td>
      <td>47.4</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>7593354</th>
      <td>2022-12-31 23:00:00</td>
      <td>23</td>
      <td>5</td>
      <td>1</td>
      <td>34200</td>
      <td>23</td>
      <td>6355.88</td>
      <td>225.461986</td>
      <td>122.57</td>
      <td>20947</td>
      <td>671</td>
      <td>2.5</td>
      <td>47.0</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>0.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 검증데이터
df_val.head()
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
      <th>electric_train.tm</th>
      <th>electric_train.hh24</th>
      <th>electric_train.weekday</th>
      <th>electric_train.week_name</th>
      <th>electric_train.elec</th>
      <th>electric_train.num</th>
      <th>electric_train.stn</th>
      <th>electric_train.nph_ta</th>
      <th>electric_train.nph_hm</th>
      <th>electric_train.nph_ws_10m</th>
      <th>electric_train.nph_rn_60m</th>
      <th>electric_train.nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35064</th>
      <td>2023-01-01</td>
      <td>24</td>
      <td>6</td>
      <td>1</td>
      <td>99.64</td>
      <td>5565</td>
      <td>184</td>
      <td>4.8</td>
      <td>66.9</td>
      <td>2.9</td>
      <td>0.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>61368</th>
      <td>2023-01-01</td>
      <td>24</td>
      <td>6</td>
      <td>1</td>
      <td>104.26</td>
      <td>5566</td>
      <td>184</td>
      <td>4.8</td>
      <td>66.9</td>
      <td>2.9</td>
      <td>0.0</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>96432</th>
      <td>2023-01-01</td>
      <td>24</td>
      <td>6</td>
      <td>1</td>
      <td>107.96</td>
      <td>8994</td>
      <td>261</td>
      <td>-3.1</td>
      <td>91.7</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>-3.1</td>
    </tr>
    <tr>
      <th>140280</th>
      <td>2023-01-01</td>
      <td>24</td>
      <td>6</td>
      <td>1</td>
      <td>111.51</td>
      <td>9736</td>
      <td>774</td>
      <td>-1.4</td>
      <td>86.8</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>-1.4</td>
    </tr>
    <tr>
      <th>166584</th>
      <td>2023-01-01</td>
      <td>24</td>
      <td>6</td>
      <td>1</td>
      <td>106.43</td>
      <td>9758</td>
      <td>168</td>
      <td>3.9</td>
      <td>68.7</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>-2.2</td>
    </tr>
  </tbody>
</table>
</div>



## 훈련 Features와 정답 label 분리 


```python
target = 'electric_train.elec'
```


```python
train_X = df_train.drop(target, axis=1)
train_Y = df_train[target]

val_X = df_val.drop(target, axis=1)
val_Y = df_val[target]

```


```python

```

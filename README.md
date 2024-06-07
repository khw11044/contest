# 2024 날씨 빅데이터 콘테스트 

## 전력 분야 - 기상에 따른 공동주택 전력수요 예측 개선 


```python
import pandas as pd
import os 

import warnings
warnings.filterwarnings(action='ignore')
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

기상 데이터 

![image](https://github.com/khw11044/csv_dataset/assets/51473705/cc9f5973-b068-4c77-812a-dc68160cd285)


```python
elec_cols = ['electric_train.'+ a for a in ['tm', 'hh24', 'weekday', 'week_name', 'sum_qctr', 'n', 'sum_load', 'n_mean_load', 'elec']]

weat_cols = ['electric_train.'+ a for a in ['num', 'stn', 'nph_ta', 'nph_hm', 'nph_ws_10m', 'nph_rn_60m', 'nph_ta_chi']]

reset_order_cols = elec_cols + weat_cols

df_new = df[reset_order_cols]
colunms = {}
for col in reset_order_cols:
    colunms[col] = col.split('.')[1]

df_new = df_new.rename(columns=colunms)

df_new.head()
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
      <th>tm</th>
      <th>hh24</th>
      <th>weekday</th>
      <th>week_name</th>
      <th>sum_qctr</th>
      <th>n</th>
      <th>sum_load</th>
      <th>n_mean_load</th>
      <th>elec</th>
      <th>num</th>
      <th>stn</th>
      <th>nph_ta</th>
      <th>nph_hm</th>
      <th>nph_ws_10m</th>
      <th>nph_rn_60m</th>
      <th>nph_ta_chi</th>
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

$\text{elec}_{i} = \frac{\text{(sum load)}_{i}}{\text{(All mean load for a year)}}$

### **활용법**

전일 또는 지난주 대비 지수의 증감비율을 통해 전력수요 변화량 예상 

**예시)** 
- 전일 최고 전력기상지수가 100, 
- 당일 최고 전력기상지수가 125,
- 당일 최대수요는 전일대비 125/100 = 1.25배 (25%) 증가를 예상해 활용 


$\text{당일 최대수요 증감소율}_{i} = \frac{\text{당일 최고 전력기상지수}(\text{max}(elec_{i}))}{\text{전일 최고 전력기상지수}(\text{max}(elec_{i-1}))}$

- elec : 전력기상지수 
- sum_qctr : 계약전력합계        = 해당격자의 전력통계 산출에 포함된 공동주택의 계약전력 합계 
- n : 공동주택 수                = 해당격자의 전력통계 산출에 포함된 공동주택의 수, 단위(단자)
- sum_load : 전력수요 합계       = 해당격자/시각에 측정된 공동주택의 전력수요 합계 
- n_mean_load : 전력부하량 평균  = 격자내 총 전력부하량을 아파트 수로 나누어 격자의 평균 부하량을 산출

$ \text{n\_mean\_load} = \text{sum\_load} \div \text{n}$

### **검증 데이터**
**검증 데이터**는 전력기상지수를 산출할 수 있는 변수인 sum_qctr(계약전력합계), n(공동주택 수), sum_load(전력수요 합계), n_mean_load(전력부하량 평균)를 제외하고 제공 

$\text{elec}_{i} = \frac{\text{(sum load)}_{i}}{\text{(n mean load)}_{i-1}}$


즉, elec(전력기상지수, 예측대상)를 구하려면, 해당격자의 해당시간 전력 수요(sum_load)를 알아야하고, 해당년도의 전체 시간 평균 전력 수요를 알아야 한다

## **목표**
공동주택 전력수요 예측 

1. 공공데이터를 활용하여 공동주택 전력수요 증감 영향 요인 분석 
2. 계절, 지역에 따른 모델 세분화를 통한 공동주택 전력 수요 예측 (전력기상지수(electric_train.elec)) 최적모델 개발 

## -------------

### AWS 넘버별 지역이 같다 라는 추측 증명 

같은 AWS 넘버별 지역은 온도와 습도등 기상 기록이 같을 것이다.


```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

```


```python
df_2021 = df_new[df_new['tm'].dt.year == 2021]
```


```python
stn_nums = []
most = 0
for stn in df_2021['stn'].unique():
    
    nums_num = len(df_2021[df_2021['stn']==stn]['num'].unique())
    stn_nums.append([stn,nums_num])
    

stn_nums.sort(key=lambda x:x[1],reverse=True)

print(stn_nums)
```

    [[846, 6], [133, 5], [541, 5], [937, 4], [899, 4], [152, 4], [146, 4], [827, 4], [138, 4], [279, 4], [493, 4], [617, 4], [550, 4], [119, 4], [565, 4], [572, 4], [511, 4], [168, 3], [162, 3], [788, 3], [908, 3], [783, 3], [255, 3], [253, 3], [941, 3], [940, 3], [143, 3], [636, 3], [371, 3], [445, 3], [114, 3], [546, 3], [404, 3], [428, 3], [450, 3], [532, 3], [184, 2], [165, 2], [774, 2], [313, 2], [710, 2], [950, 2], [904, 2], [942, 2], [156, 2], [192, 2], [673, 2], [939, 2], [257, 2], [864, 2], [860, 2], [702, 2], [840, 2], [140, 2], [845, 2], [824, 2], [822, 2], [642, 2], [496, 2], [131, 2], [327, 2], [693, 2], [129, 2], [551, 2], [127, 2], [545, 2], [512, 2], [434, 2], [590, 2], [112, 2], [649, 2], [376, 2], [543, 2], [433, 2], [427, 2], [589, 2], [506, 2], [599, 2], [353, 2], [101, 2], [884, 1], [330, 1], [261, 1], [294, 1], [712, 1], [713, 1], [266, 1], [907, 1], [917, 1], [708, 1], [722, 1], [159, 1], [155, 1], [938, 1], [974, 1], [252, 1], [923, 1], [247, 1], [288, 1], [245, 1], [898, 1], [901, 1], [900, 1], [284, 1], [943, 1], [737, 1], [991, 1], [811, 1], [281, 1], [825, 1], [615, 1], [605, 1], [235, 1], [643, 1], [137, 1], [612, 1], [494, 1], [611, 1], [273, 1], [136, 1], [177, 1], [628, 1], [627, 1], [837, 1], [616, 1], [600, 1], [472, 1], [358, 1], [495, 1], [516, 1], [471, 1], [432, 1], [446, 1], [221, 1], [571, 1], [216, 1], [436, 1], [549, 1], [203, 1], [430, 1], [533, 1], [548, 1], [438, 1], [364, 1], [459, 1], [377, 1], [492, 1], [509, 1], [423, 1], [417, 1], [410, 1], [889, 1], [401, 1], [400, 1], [403, 1], [202, 1], [876, 1], [405, 1], [418, 1], [415, 1], [421, 1], [413, 1], [402, 1], [441, 1], [412, 1], [108, 1], [408, 1], [409, 1], [569, 1], [106, 1], [570, 1], [540, 1], [416, 1], [414, 1], [424, 1], [407, 1], [406, 1], [484, 1], [373, 1], [104, 1], [524, 1], [99, 1], [98, 1], [671, 1]]
    


```python
df_temp = df_2021[df_2021['stn']==846]
```


```python

fig = plt.figure(figsize = (15, 5))
for idx, num in enumerate(df_temp['num'].unique()):
    ax = plt.subplot(2, 5, 1+idx)
    energy = df_temp['nph_ta'].values

    plt.hist(energy, alpha = 0.7, bins = 50, color = 'gray')
    
    num = df_temp['num'].unique()[idx]
    plt.title(f'격자넘버: {num}')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
plt.tight_layout()
plt.show()
```


    
![png](1.data_analysis_1_files/1.data_analysis_1_15_0.png)
    



```python
check_cols = ['stn', 'nph_ta', 'nph_hm', 'nph_ws_10m', 'nph_rn_60m', 'nph_ta_chi']

df_temp[(df_temp['num']==13199) & (df_temp.tm=='2021-01-01 01:00:00')][check_cols].iloc[:2]
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
      <th>stn</th>
      <th>nph_ta</th>
      <th>nph_hm</th>
      <th>nph_ws_10m</th>
      <th>nph_rn_60m</th>
      <th>nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2121912</th>
      <td>846</td>
      <td>2.3</td>
      <td>62.4</td>
      <td>5.3</td>
      <td>0.0</td>
      <td>-7.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_temp[(df_temp['num']==13200) & (df_temp.tm=='2021-01-01 01:00:00')][check_cols].iloc[:2]
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
      <th>stn</th>
      <th>nph_ta</th>
      <th>nph_hm</th>
      <th>nph_ws_10m</th>
      <th>nph_rn_60m</th>
      <th>nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2148216</th>
      <td>846</td>
      <td>2.3</td>
      <td>62.4</td>
      <td>5.3</td>
      <td>0.0</td>
      <td>-7.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_temp[(df_temp['num']==13348) & (df_temp.tm=='2021-01-01 01:00:00')][check_cols].iloc[:2]
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
      <th>stn</th>
      <th>nph_ta</th>
      <th>nph_hm</th>
      <th>nph_ws_10m</th>
      <th>nph_rn_60m</th>
      <th>nph_ta_chi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2306040</th>
      <td>846</td>
      <td>2.3</td>
      <td>62.4</td>
      <td>5.3</td>
      <td>0.0</td>
      <td>-7.2</td>
    </tr>
  </tbody>
</table>
</div>



## Pre-processing & EDA


```python
df_new['year'] = df_new['tm'].dt.year
df_new['month'] = df_new['tm'].dt.month
df_new['day'] = df_new['tm'].dt.day
df_new = df_new.sort_values(by='tm')

# 시즌을 결정하는 함수
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

# 'season' 컬럼 추가
df_new['season'] = df_new['tm'].dt.month.apply(get_season)
df_new['season'] = df_new['season'].map({'Spring':0, 'Summer':1, 'Fall':2, 'Winter':3})

# 공휴일을 주말 카테고리에 추가
from pytimekr import pytimekr
kr_holidays_2021 = pytimekr.holidays(year=2021)
kr_holidays_2022 = pytimekr.holidays(year=2022)
kr_holidays_2023 = pytimekr.holidays(year=2023)

df_new.loc[df_new['tm'].isin(kr_holidays_2021),'week_name']=1
df_new.loc[df_new['tm'].isin(kr_holidays_2022),'week_name']=1
df_new.loc[df_new['tm'].isin(kr_holidays_2023),'week_name']=1

# 시간의 연속성 추가
# ## sin cos 함수를 이용한 시간의 연속적 표현 (cyclical time encoding)
import numpy as np
df_new['sin_time'] = np.sin(2*np.pi*df_new.hh24/24)
df_new['cos_time'] = np.cos(2*np.pi*df_new.hh24/24)

# THI(불쾌지수) & CDH(Cooling Degree Hour) 변수 추가
df_new['THI'] = 9/5*df_new['nph_ta'] - 0.55*(1-df_new['nph_hm']/100)*(9/5*df_new['nph_hm']-26)+32
def CDH(xs):
    ys = []
    for i in range(len(xs)):
        if i < 11:
            ys.append(np.sum(xs[:(i+1)]-26))        # 26도 
        else:
            ys.append(np.sum(xs[(i-11):(i+1)]-26))
    return np.array(ys)

df_new['CDH'] = 0
cdhs = np.array([])
for num in df_new['num'].unique():                   # 격자별 불쾌지수 
    temp = df_new[df_new['num'] == num]
    cdh = CDH(temp['nph_ta'].values)
    df_new.loc[df_new['num'] == num, 'CDH'] = cdh

cols_for_test = [
        'tm', 'year', 'season', 'month','day', 'hh24', 'weekday','week_name', 'sin_time', 'cos_time',
        'num',
        'stn', 'nph_ta','nph_hm', 'CDH', 'THI',
        'nph_ws_10m', 'nph_rn_60m', 'nph_ta_chi', 
       'elec']

df_train = df_new[cols_for_test]

df_train.head()
```

    C:\Users\kim_h\AppData\Local\Temp\ipykernel_6408\1157259303.py:27: FutureWarning: The behavior of 'isin' with dtype=datetime64[ns] and castable values (e.g. strings) is deprecated. In a future version, these will not be considered matching by isin. Explicitly cast to the appropriate dtype before calling isin instead.
      df_new.loc[df_new['tm'].isin(kr_holidays_2021),'week_name']=1
    C:\Users\kim_h\AppData\Local\Temp\ipykernel_6408\1157259303.py:28: FutureWarning: The behavior of 'isin' with dtype=datetime64[ns] and castable values (e.g. strings) is deprecated. In a future version, these will not be considered matching by isin. Explicitly cast to the appropriate dtype before calling isin instead.
      df_new.loc[df_new['tm'].isin(kr_holidays_2022),'week_name']=1
    C:\Users\kim_h\AppData\Local\Temp\ipykernel_6408\1157259303.py:29: FutureWarning: The behavior of 'isin' with dtype=datetime64[ns] and castable values (e.g. strings) is deprecated. In a future version, these will not be considered matching by isin. Explicitly cast to the appropriate dtype before calling isin instead.
      df_new.loc[df_new['tm'].isin(kr_holidays_2023),'week_name']=1
    C:\Users\kim_h\AppData\Local\Temp\ipykernel_6408\1157259303.py:53: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '[ -34.5  -69.5 -104.2 ... -311.  -317.2 -325.9]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
      df_new.loc[df_new['num'] == num, 'CDH'] = cdh
    




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
      <th>tm</th>
      <th>year</th>
      <th>season</th>
      <th>month</th>
      <th>day</th>
      <th>hh24</th>
      <th>weekday</th>
      <th>week_name</th>
      <th>sin_time</th>
      <th>cos_time</th>
      <th>num</th>
      <th>stn</th>
      <th>nph_ta</th>
      <th>nph_hm</th>
      <th>CDH</th>
      <th>THI</th>
      <th>nph_ws_10m</th>
      <th>nph_rn_60m</th>
      <th>nph_ta_chi</th>
      <th>elec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2647991</th>
      <td>2020-01-01 01:00:00</td>
      <td>2020</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.258819</td>
      <td>0.965926</td>
      <td>13615</td>
      <td>140</td>
      <td>-8.5</td>
      <td>74.5</td>
      <td>-34.5</td>
      <td>1.538975</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>-5.8</td>
      <td>99.63</td>
    </tr>
    <tr>
      <th>6576212</th>
      <td>2020-01-01 01:00:00</td>
      <td>2020</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.258819</td>
      <td>0.965926</td>
      <td>18837</td>
      <td>569</td>
      <td>-3.5</td>
      <td>22.8</td>
      <td>-29.5</td>
      <td>19.314016</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-5.6</td>
      <td>98.77</td>
    </tr>
    <tr>
      <th>6602516</th>
      <td>2020-01-01 01:00:00</td>
      <td>2020</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.258819</td>
      <td>0.965926</td>
      <td>18838</td>
      <td>428</td>
      <td>-2.3</td>
      <td>21.2</td>
      <td>-28.3</td>
      <td>22.589856</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>-6.1</td>
      <td>99.56</td>
    </tr>
    <tr>
      <th>543649</th>
      <td>2020-01-01 01:00:00</td>
      <td>2020</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.258819</td>
      <td>0.965926</td>
      <td>10824</td>
      <td>950</td>
      <td>2.9</td>
      <td>67.8</td>
      <td>-23.1</td>
      <td>20.211316</td>
      <td>1.1</td>
      <td>0.0</td>
      <td>-5.9</td>
      <td>98.32</td>
    </tr>
    <tr>
      <th>6628820</th>
      <td>2020-01-01 01:00:00</td>
      <td>2020</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.258819</td>
      <td>0.965926</td>
      <td>18870</td>
      <td>106</td>
      <td>-1.8</td>
      <td>25.5</td>
      <td>-27.8</td>
      <td>20.605975</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>-6.3</td>
      <td>101.20</td>
    </tr>
  </tbody>
</table>
</div>



### 이상치 처리


```python
import seaborn as sns

# 온도별 elec에서 elec 이상치 발견
graph = sns.jointplot(x=df_train['nph_ta'], y=df_train['elec'], kind='scatter')
graph.set_axis_labels(xlabel='Temperature', ylabel='elec')
plt.show()
```


    
![png](1.data_analysis_1_files/1.data_analysis_1_22_0.png)
    



```python
# 풍속별 elec에서 풍속 이상치 발견
graph = sns.jointplot(x=df_train['nph_ws_10m'], y=df_train['elec'], kind='scatter')
graph.set_axis_labels(xlabel='Wind', ylabel='elec')
plt.show()
```


    
![png](1.data_analysis_1_files/1.data_analysis_1_23_0.png)
    



```python
# 온도별 elec의 이상치 처리 
for idx in list(df_train.loc[df_train['elec'] < 0, 'nph_ta'].index):
    area_num = df_train.loc[idx]['num']
    season = df_train.loc[idx]['season']
    year = df_train.loc[idx]['year']
    temp = df_train.loc[idx]['nph_ta']
    df_train.loc[idx, 'elec'] = \
        df_train.loc[(df_train.loc[idx]['num'] == area_num) & (df_train.loc[idx]['year'] == year) & (df_train.loc[idx]['season'] == season) & (df_train['nph_ta'] == temp), :]['elec'].mean()
        # 해당 지역의 해당 연도의 해당 시즌의 해당 온도와 같은 날의 elec의 평균
```


```python
# 풍속이 음수인 값을 갖는 이상치 처리

ano_list = list(df_train.loc[df_train['nph_ws_10m'] < 0, 'nph_ws_10m'].index)
for idx in ano_list:
    pre_idx = idx - 1
    nex_idx = idx + 1

    previous_value = df_train.loc[pre_idx, 'nph_ws_10m']
    next_value = df_train.loc[nex_idx, 'nph_ws_10m']
    mean_val = (previous_value + next_value) / 2
    df_train.loc[idx, 'nph_ws_10m'] = mean_val
```


```python
## plot feature data distribution

fig, ax = plt.subplots(2, df_train.shape[1]//2+1, figsize=(20, 6))
for idx, feature in enumerate(df_train.columns):
    data = df_train[feature]
    if idx<df_train.shape[1]//2 + 1:
        ax[0,idx].hist(df_train.iloc[:,idx], bins=10, alpha=0.5)
        ax[0,idx].set_title(df_train.columns[idx])
    else:
        ax[1,idx-df_train.shape[1]//2-1].hist(df_train.iloc[:,idx], bins=10, alpha=0.5)
        ax[1,idx-df_train.shape[1]//2-1].set_title(df_train.columns[idx])
        
plt.tight_layout()
plt.show()
```


    
![png](1.data_analysis_1_files/1.data_analysis_1_26_0.png)
    



```python
## chk feature correlation visually

fig, axes = plt.subplots(3, 4, figsize=(10,7))
df_train.plot(x='year', y='elec', kind='scatter', alpha=0.1, ax=axes[0,0])
df_train.plot(x='month', y='elec', kind='scatter', alpha=0.1, ax=axes[0,1])
df_train.plot(x='weekday', y='elec', kind='scatter', alpha=0.1, ax=axes[0,2])
df_train.plot(x='hh24', y='elec', kind='scatter', alpha=0.1, ax=axes[0,3])

df_train.plot(x='nph_ta', y='elec', kind='scatter', alpha=0.1, ax=axes[1,0])
df_train.plot(x='nph_hm', y='elec', kind='scatter', alpha=0.1, ax=axes[1,1])
df_train.plot(x='CDH', y='elec', kind='scatter', alpha=0.1, ax=axes[1,2])
df_train.plot(x='THI', y='elec', kind='scatter', alpha=0.1, ax=axes[1,3])

df_train.plot(x='nph_ws_10m', y='elec', kind='scatter', alpha=0.1, ax=axes[2,0])
df_train.plot(x='nph_rn_60m', y='elec', kind='scatter', alpha=0.1, ax=axes[2,1])
df_train.plot(x='nph_ta_chi', y='elec', kind='scatter', alpha=0.1, ax=axes[2,2])

fig.tight_layout()
```


    
![png](1.data_analysis_1_files/1.data_analysis_1_27_0.png)
    



```python
# 상관관계 시각화 
plt.figure(figsize=(12,12))
sns.heatmap(df_train.corr(numeric_only=True),
           annot=True,
           cmap='Blues',
           cbar=False, # 옆에 칼라 바 제거 
           square=True,
            fmt='.3f', # 소수점
            annot_kws={'size':9}
           )    
plt.show()
```


    
![png](1.data_analysis_1_files/1.data_analysis_1_28_0.png)
    



```python

```

# Workshop#04_Customer Scoring and Churn

## 	:point_right: Churn Prediction คืออะไร?
**Churn Prediction หรือ การทำนายลักษณะลูกค้าที่กำลังจะยกเลิกบริการ** เป็นวิเคราะห์ข้อมูลพฤติกรรมของลูกค้าที่มีแนวโน้มจะยกเลิกการใช้บริการ/ผลิตภัณฑ์ของบริษัท รวมถึงหาข้อบ่งชี้ที่ส่งผลต่อ Churn rate ของลูกค้า

**_ยกตัวอย่างเช่น:_**
* พฤติกรรมที่การซื้อเปลี่ยนไป เช่น ซื้อผลิตภัณฑ์อื่น, ซื้อน้อยลงกว่าเดิมหรือไม่ซื้ออีกเลย, ระยะเวลาในการกลับมาซื้อนานขึ้นเรื่อย ๆ
* ลูกค้าไม่พอใจในสินค้าหรือบริการ เช่น มีการร้องเรียน ส่งคืนสินค้า หรือให้คะแนน/รีวิวไม่ดี


## 	:point_right: Churn Prediction สำคัญอย่างไร?
* ทำให้บริษัทสามารถวิเคราะห์ได้ว่าลูกค้า Churn เพราะปัจจัยใด และหากลยุทธ์/แนวทางเพื่อลดโอกาสในการ Chrun ของลูกค้า
* สามารถวิเคราะห์ได้ว่าช่วงเวลาใดที่ลูกค้ามีโอกาส Churn สูง และรีบเข้าถึงลูกค้า (เช่น ยิง Ads โฆษณา, ปล่อย promotion) ก่อนที่จะถึงระยะเวลานั้น

![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/7ae3e643-e12f-401c-a8a4-8e6bbb14f7d8)

_ภาพจาก: https://www.linkedin.com/pulse/3-ways-predict-your-customer-churn-plytrix-analytics/_

## 	:point_right: Machine Learning for Churn Prediction
ตัวอย่างการใช้งาน Machine Learning เพื่อใช้ Churn prediction สามารถใช้ได้หลาย Algorithm ขึ้นอยู่กับปัจจัย หรือตัวแปรที่เราต้องการนำมาวิเคราะห์ เช่น
* Logistic Regression
* Random Forest Classifier
* K-Nearest Neighbors Classifier
* SMOTE
* XGBoost Classifier

ซึ่งในวันนี้เราจะมาลองทำ Churn Prediction โดยใช้ Python ตามรายละเอียดด้านล่าง

### :white_check_mark: Dataset
[ecommerce_Dataset.csv](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2304_Customer%20Scoring%20and%20Churn/ecommerce_Dataset.csv)

### :white_check_mark: Import Libraries and Data
```
## Import Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

```
## Import Data
df = pd.read_csv("ecommerce_Dataset.csv")
```

### :white_check_mark: Exploratory Data Analysis (EDA)
```
## Explore Data
df.head()
```

```
df.info()
```

```
df.isna().sum()
```

```
df.describe()
```
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/39e32b2a-1bd5-41aa-a74e-e866d17f9478)

> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/4d399cc1-1a8a-47e3-896c-48273f9312de)

> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/9b5c9e23-7721-4b32-9d68-fae2c115d155)



### :white_check_mark: Model Evaluation

|                      Model                       |   Sampling Method    | Precision (0) | Recall (0) | F1-Score (0) | Support (0) | Precision (1) | Recall (1) | F1-Score (1) | Support (1) | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score | Macro Avg Support | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1-Score | Weighted Avg Support |
|--------------------------------------------------|----------------------|---------------|------------|--------------|-------------|---------------|------------|--------------|-------------|----------|---------------------|------------------|--------------------|-------------------|------------------------|---------------------|-----------------------|----------------------|
|               Logistic Regression                |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.67      |    0.41    |     0.51     |   246.00    |   0.86   |        0.78         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.86         |         0.85          |       1408.00        |
|               Logistic Regression                |        SMOTE         |     0.93      |    0.72    |     0.81     |   1162.00   |     0.36      |    0.76    |     0.49     |   246.00    |   0.72   |        0.65         |       0.74       |        0.65        |      1408.00      |          0.83          |        0.72         |         0.75          |       1408.00        |
|               Logistic Regression                | Random Oversampling  |     0.95      |    0.75    |     0.84     |   1162.00   |     0.40      |    0.80    |     0.54     |   246.00    |   0.76   |        0.68         |       0.77       |        0.69        |      1408.00      |          0.85          |        0.76         |         0.79          |       1408.00        |
|               Logistic Regression                | Random Undersampling |     0.95      |    0.74    |     0.83     |   1162.00   |     0.40      |    0.80    |     0.53     |   246.00    |   0.75   |        0.67         |       0.77       |        0.68        |      1408.00      |          0.85          |        0.75         |         0.78          |       1408.00        |
|             Random Forest Classifier             |     No Sampling      |     0.97      |    0.99    |     0.98     |   1162.00   |     0.97      |    0.85    |     0.90     |   246.00    |   0.97   |        0.97         |       0.92       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|             Random Forest Classifier             |        SMOTE         |     0.97      |    0.99    |     0.98     |   1162.00   |     0.93      |    0.85    |     0.89     |   246.00    |   0.96   |        0.95         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|             Random Forest Classifier             | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.96      |    0.86    |     0.91     |   246.00    |   0.97   |        0.97         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|             Random Forest Classifier             | Random Undersampling |     0.98      |    0.90    |     0.94     |   1162.00   |     0.66      |    0.90    |     0.76     |   246.00    |   0.90   |        0.82         |       0.90       |        0.85        |      1408.00      |          0.92          |        0.90         |         0.91          |       1408.00        |
|          K Nearest Neighbors Classifier          |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.69      |    0.42    |     0.52     |   246.00    |   0.87   |        0.79         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.87         |         0.85          |       1408.00        |
|          K Nearest Neighbors Classifier          |        SMOTE         |     0.97      |    0.79    |     0.87     |   1162.00   |     0.47      |    0.87    |     0.61     |   246.00    |   0.80   |        0.72         |       0.83       |        0.74        |      1408.00      |          0.88          |        0.80         |         0.82          |       1408.00        |
|          K Nearest Neighbors Classifier          | Random Oversampling  |     0.97      |    0.84    |     0.90     |   1162.00   |     0.52      |    0.86    |     0.65     |   246.00    |   0.84   |        0.75         |       0.85       |        0.77        |      1408.00      |          0.89          |        0.84         |         0.85          |       1408.00        |
|          K Nearest Neighbors Classifier          | Random Undersampling |     0.94      |    0.72    |     0.81     |   1162.00   |     0.37      |    0.78    |     0.50     |   246.00    |   0.73   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.73         |         0.76          |       1408.00        |
|          SMOTE and Logistic Regression           |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.67      |    0.41    |     0.51     |   246.00    |   0.86   |        0.78         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.86         |         0.85          |       1408.00        |
|          SMOTE and Logistic Regression           |        SMOTE         |     0.93      |    0.72    |     0.81     |   1162.00   |     0.36      |    0.76    |     0.49     |   246.00    |   0.72   |        0.65         |       0.74       |        0.65        |      1408.00      |          0.83          |        0.72         |         0.75          |       1408.00        |
|          SMOTE and Logistic Regression           | Random Oversampling  |     0.95      |    0.75    |     0.84     |   1162.00   |     0.40      |    0.80    |     0.54     |   246.00    |   0.76   |        0.68         |       0.77       |        0.69        |      1408.00      |          0.85          |        0.76         |         0.79          |       1408.00        |
|          SMOTE and Logistic Regression           | Random Undersampling |     0.95      |    0.74    |     0.83     |   1162.00   |     0.40      |    0.80    |     0.53     |   246.00    |   0.75   |        0.67         |       0.77       |        0.68        |      1408.00      |          0.85          |        0.75         |         0.78          |       1408.00        |
|        SMOTE and Random Forest Classifier        |     No Sampling      |     0.96      |    0.99    |     0.98     |   1162.00   |     0.97      |    0.83    |     0.89     |   246.00    |   0.97   |        0.97         |       0.91       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.96          |       1408.00        |
|        SMOTE and Random Forest Classifier        |        SMOTE         |     0.96      |    0.98    |     0.97     |   1162.00   |     0.91      |    0.82    |     0.87     |   246.00    |   0.96   |        0.94         |       0.90       |        0.92        |      1408.00      |          0.95          |        0.96         |         0.95          |       1408.00        |
|        SMOTE and Random Forest Classifier        | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.94      |    0.87    |     0.91     |   246.00    |   0.97   |        0.96         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|        SMOTE and Random Forest Classifier        | Random Undersampling |     0.98      |    0.90    |     0.93     |   1162.00   |     0.65      |    0.90    |     0.75     |   246.00    |   0.90   |        0.81         |       0.90       |        0.84        |      1408.00      |          0.92          |        0.90         |         0.90          |       1408.00        |
|     SMOTE and K Nearest Neighbors Classifier     |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.69      |    0.42    |     0.52     |   246.00    |   0.87   |        0.79         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.87         |         0.85          |       1408.00        |
|     SMOTE and K Nearest Neighbors Classifier     |        SMOTE         |     0.97      |    0.79    |     0.87     |   1162.00   |     0.47      |    0.87    |     0.61     |   246.00    |   0.80   |        0.72         |       0.83       |        0.74        |      1408.00      |          0.88          |        0.80         |         0.82          |       1408.00        |
|     SMOTE and K Nearest Neighbors Classifier     | Random Oversampling  |     0.97      |    0.84    |     0.90     |   1162.00   |     0.52      |    0.86    |     0.65     |   246.00    |   0.84   |        0.75         |       0.85       |        0.77        |      1408.00      |          0.89          |        0.84         |         0.85          |       1408.00        |
|     SMOTE and K Nearest Neighbors Classifier     | Random Undersampling |     0.94      |    0.72    |     0.81     |   1162.00   |     0.37      |    0.78    |     0.50     |   246.00    |   0.73   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.73         |         0.76          |       1408.00        |
|       Oversampling and Logistic Regression       |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.67      |    0.41    |     0.51     |   246.00    |   0.86   |        0.78         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.86         |         0.85          |       1408.00        |
|       Oversampling and Logistic Regression       |        SMOTE         |     0.93      |    0.72    |     0.81     |   1162.00   |     0.36      |    0.76    |     0.49     |   246.00    |   0.72   |        0.65         |       0.74       |        0.65        |      1408.00      |          0.83          |        0.72         |         0.75          |       1408.00        |
|       Oversampling and Logistic Regression       | Random Oversampling  |     0.95      |    0.75    |     0.84     |   1162.00   |     0.40      |    0.80    |     0.54     |   246.00    |   0.76   |        0.68         |       0.77       |        0.69        |      1408.00      |          0.85          |        0.76         |         0.79          |       1408.00        |
|       Oversampling and Logistic Regression       | Random Undersampling |     0.95      |    0.74    |     0.83     |   1162.00   |     0.40      |    0.80    |     0.53     |   246.00    |   0.75   |        0.67         |       0.77       |        0.68        |      1408.00      |          0.85          |        0.75         |         0.78          |       1408.00        |
|    Oversampling and Random Forest Classifier     |     No Sampling      |     0.96      |    1.00    |     0.98     |   1162.00   |     0.98      |    0.83    |     0.90     |   246.00    |   0.97   |        0.97         |       0.91       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|    Oversampling and Random Forest Classifier     |        SMOTE         |     0.97      |    0.98    |     0.97     |   1162.00   |     0.92      |    0.83    |     0.87     |   246.00    |   0.96   |        0.94         |       0.91       |        0.92        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|    Oversampling and Random Forest Classifier     | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.96      |    0.87    |     0.91     |   246.00    |   0.97   |        0.96         |       0.93       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|    Oversampling and Random Forest Classifier     | Random Undersampling |     0.98      |    0.89    |     0.93     |   1162.00   |     0.64      |    0.91    |     0.75     |   246.00    |   0.90   |        0.81         |       0.90       |        0.84        |      1408.00      |          0.92          |        0.90         |         0.90          |       1408.00        |
| Oversampling and K Nearest Neighbors Classifier  |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.69      |    0.42    |     0.52     |   246.00    |   0.87   |        0.79         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.87         |         0.85          |       1408.00        |
| Oversampling and K Nearest Neighbors Classifier  |        SMOTE         |     0.97      |    0.79    |     0.87     |   1162.00   |     0.47      |    0.87    |     0.61     |   246.00    |   0.80   |        0.72         |       0.83       |        0.74        |      1408.00      |          0.88          |        0.80         |         0.82          |       1408.00        |
| Oversampling and K Nearest Neighbors Classifier  | Random Oversampling  |     0.97      |    0.84    |     0.90     |   1162.00   |     0.52      |    0.86    |     0.65     |   246.00    |   0.84   |        0.75         |       0.85       |        0.77        |      1408.00      |          0.89          |        0.84         |         0.85          |       1408.00        |
| Oversampling and K Nearest Neighbors Classifier  | Random Undersampling |     0.94      |    0.72    |     0.81     |   1162.00   |     0.37      |    0.78    |     0.50     |   246.00    |   0.73   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.73         |         0.76          |       1408.00        |
|      Undersampling and Logistic Regression       |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.67      |    0.41    |     0.51     |   246.00    |   0.86   |        0.78         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.86         |         0.85          |       1408.00        |
|      Undersampling and Logistic Regression       |        SMOTE         |     0.93      |    0.72    |     0.81     |   1162.00   |     0.36      |    0.76    |     0.49     |   246.00    |   0.72   |        0.65         |       0.74       |        0.65        |      1408.00      |          0.83          |        0.72         |         0.75          |       1408.00        |
|      Undersampling and Logistic Regression       | Random Oversampling  |     0.95      |    0.75    |     0.84     |   1162.00   |     0.40      |    0.80    |     0.54     |   246.00    |   0.76   |        0.68         |       0.77       |        0.69        |      1408.00      |          0.85          |        0.76         |         0.79          |       1408.00        |
|      Undersampling and Logistic Regression       | Random Undersampling |     0.95      |    0.74    |     0.83     |   1162.00   |     0.40      |    0.80    |     0.53     |   246.00    |   0.75   |        0.67         |       0.77       |        0.68        |      1408.00      |          0.85          |        0.75         |         0.78          |       1408.00        |
|    Undersampling and Random Forest Classifier    |     No Sampling      |     0.96      |    0.99    |     0.98     |   1162.00   |     0.94      |    0.83    |     0.88     |   246.00    |   0.96   |        0.95         |       0.91       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|    Undersampling and Random Forest Classifier    |        SMOTE         |     0.97      |    0.98    |     0.97     |   1162.00   |     0.90      |    0.85    |     0.87     |   246.00    |   0.96   |        0.93         |       0.91       |        0.92        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|    Undersampling and Random Forest Classifier    | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.97      |    0.87    |     0.91     |   246.00    |   0.97   |        0.97         |       0.93       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|    Undersampling and Random Forest Classifier    | Random Undersampling |     0.98      |    0.89    |     0.93     |   1162.00   |     0.63      |    0.90    |     0.74     |   246.00    |   0.89   |        0.80         |       0.89       |        0.83        |      1408.00      |          0.92          |        0.89         |         0.90          |       1408.00        |
| Undersampling and K Nearest Neighbors Classifier |     No Sampling      |     0.89      |    0.96    |     0.92     |   1162.00   |     0.69      |    0.42    |     0.52     |   246.00    |   0.87   |        0.79         |       0.69       |        0.72        |      1408.00      |          0.85          |        0.87         |         0.85          |       1408.00        |
| Undersampling and K Nearest Neighbors Classifier |        SMOTE         |     0.97      |    0.79    |     0.87     |   1162.00   |     0.47      |    0.87    |     0.61     |   246.00    |   0.80   |        0.72         |       0.83       |        0.74        |      1408.00      |          0.88          |        0.80         |         0.82          |       1408.00        |
| Undersampling and K Nearest Neighbors Classifier | Random Oversampling  |     0.97      |    0.84    |     0.90     |   1162.00   |     0.52      |    0.86    |     0.65     |   246.00    |   0.84   |        0.75         |       0.85       |        0.77        |      1408.00      |          0.89          |        0.84         |         0.85          |       1408.00        |
| Undersampling and K Nearest Neighbors Classifier | Random Undersampling |     0.94      |    0.72    |     0.81     |   1162.00   |     0.37      |    0.78    |     0.50     |   246.00    |   0.73   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.73         |         0.76          |       1408.00        |
|                XGBoost Classifier                |     No Sampling      |     0.97      |    1.00    |     0.98     |   1162.00   |     0.98      |    0.87    |     0.92     |   246.00    |   0.97   |        0.98         |       0.93       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|                XGBoost Classifier                |        SMOTE         |     0.97      |    0.98    |     0.98     |   1162.00   |     0.92      |    0.85    |     0.89     |   246.00    |   0.96   |        0.95         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|                XGBoost Classifier                | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.93      |    0.88    |     0.90     |   246.00    |   0.97   |        0.95         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|                XGBoost Classifier                | Random Undersampling |     0.98      |    0.90    |     0.94     |   1162.00   |     0.65      |    0.91    |     0.76     |   246.00    |   0.90   |        0.82         |       0.91       |        0.85        |      1408.00      |          0.92          |        0.90         |         0.91          |       1408.00        |
|           Oversamp XGBoost Classifier            |     No Sampling      |     0.97      |    1.00    |     0.98     |   1162.00   |     0.98      |    0.87    |     0.92     |   246.00    |   0.97   |        0.98         |       0.93       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|           Oversamp XGBoost Classifier            |        SMOTE         |     0.97      |    0.98    |     0.98     |   1162.00   |     0.92      |    0.85    |     0.89     |   246.00    |   0.96   |        0.95         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|           Oversamp XGBoost Classifier            | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.93      |    0.88    |     0.90     |   246.00    |   0.97   |        0.95         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|           Oversamp XGBoost Classifier            | Random Undersampling |     0.98      |    0.90    |     0.94     |   1162.00   |     0.65      |    0.91    |     0.76     |   246.00    |   0.90   |        0.82         |       0.91       |        0.85        |      1408.00      |          0.92          |        0.90         |         0.91          |       1408.00        |
|         Undersampled XGBoost Classifier          |     No Sampling      |     0.97      |    1.00    |     0.98     |   1162.00   |     0.98      |    0.87    |     0.92     |   246.00    |   0.97   |        0.98         |       0.93       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|         Undersampled XGBoost Classifier          |        SMOTE         |     0.97      |    0.98    |     0.98     |   1162.00   |     0.92      |    0.85    |     0.89     |   246.00    |   0.96   |        0.95         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |       1408.00        |
|         Undersampled XGBoost Classifier          | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1162.00   |     0.93      |    0.88    |     0.90     |   246.00    |   0.97   |        0.95         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |       1408.00        |
|         Undersampled XGBoost Classifier          | Random Undersampling |     0.98      |    0.90    |     0.94     |   1162.00   |     0.65      |    0.91    |     0.76     |   246.00    |   0.90   |        0.82         |       0.91       |        0.85        |      1408.00      |          0.92          |        0.90         |         0.91          |       1408.00        |


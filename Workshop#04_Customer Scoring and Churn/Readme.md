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

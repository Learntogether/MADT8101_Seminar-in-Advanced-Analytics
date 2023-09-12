# Workshop#04_Customer Scoring and Churn

## 	:point_right: Churn Prediction คืออะไร?
**Churn Prediction หรือ การทำนายลักษณะลูกค้าที่กำลังจะยกเลิกบริการ** เป็นวิเคราะห์ข้อมูลพฤติกรรมของลูกค้าที่มีแนวโน้มจะยกเลิกการใช้บริการ/ผลิตภัณฑ์ของบริษัท รวมถึงหาข้อบ่งชี้ที่ส่งผลต่อ Churn rate ของลูกค้า

**_ยกตัวอย่างเช่น:_**
* พฤติกรรมที่การซื้อเปลี่ยนไป เช่น ซื้อผลิตภัณฑ์อื่น, ซื้อน้อยลงกว่าเดิมหรือไม่ซื้ออีกเลย, ระยะเวลาในการกลับมาซื้อนานขึ้นเรื่อย ๆ
* ลูกค้าไม่พอใจในสินค้าหรือบริการ เช่น มีการร้องเรียน ส่งคืนสินค้า หรือให้คะแนน/รีวิวไม่ดี


## 	:point_right: Churn Prediction สำคัญอย่างไร?
* ทำให้บริษัทสามารถวิเคราะห์ได้ว่าลูกค้า Churn เพราะปัจจัยใด และหากลยุทธ์/แนวทางเพื่อลดโอกาสในการ Chrun ของลูกค้า
* สามารถวิเคราะห์ได้ว่าช่วงเวลาใดที่ลูกค้ามีโอกาส Churn สูง และรีบเข้าถึงลูกค้า (เช่น ยิง Ads โฆษณา, ปล่อย promotion) ก่อนที่จะถึงระยะเวลานั้น

![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/9c166677-43df-4a00-ac83-70cd493d2be2)


_ภาพจาก: https://www.linkedin.com/pulse/3-ways-predict-your-customer-churn-plytrix-analytics/_

## 	:point_right: Workshop - Churn Prediction

### :white_check_mark: Algorithm
> Example of algorithm for Churn prediction which might be different due to objective and variables:
> * Logistic Regression
> * Random Forest Classifier
> * K-Nearest Neighbors Classifier
> * SMOTE
> * XGBoost Classifier

### :white_check_mark: Dataset and Python code
> * [Ecommerce_Dataset](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2304_Customer%20Scoring%20and%20Churn/ecommerce_Dataset.csv)
> * [Python Code](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2304_Customer%20Scoring%20and%20Churn/Churn_E_Commerce.ipynb)
>
> Variables on the dataset are consist of Customer ID, Customer Demographics, Customer Account Information and Customer Feedback or purchasing behavior as follows:
> 
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/72c3dd6f-eb2a-425c-aa5f-3a55fba17467)

### :white_check_mark: Import Libraries
> Libraries: NumPy, pandas, sklearn, Matplotlib, and seaborn

### :white_check_mark: Exploratory Data Analysis (EDA) and Preparation
>
> :green_circle: **Check the Data (Mean, Min, Max, SD and Count)**
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/58c4ee65-863d-4f04-b6b6-d857052fb3f3)
>
> :green_circle: **Remove the columns that irrelevant to our needs and plot boxplot for variables that predict the customer churn**
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/72be1a68-649c-47c7-98f0-7de64a4178f7)
>
> :green_circle: **Handling Data - Missing value, Define Cat/Num variables**
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/4b1c2e0c-1514-4c7f-95ce-208815f6ec54)
>
> :green_circle: **Visualise each variable and distribution**
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/677f4e76-a4f3-4253-8e2e-472864b86839)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/ee278e3d-517a-415c-851c-4bc8b1fe7483)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/0b82c707-1a9d-42ae-be7e-bcb9c76021ab)
> 
> :green_circle: **Visualise churn with each variable and correlation coefficient**
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/6d637fce-e05b-4642-9b7e-2a94bdb1a6bf)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/7770ca63-4945-4536-a074-e217e46333ea)
>
> :green_circle: **Splitting the data into train and test sets**
>```
> X_train, X_test,y_train, y_test = train_test_split(df.drop('Churn', axis=1), df.Churn)
> ```
>
> :green_circle: **Train the Model and Prediction**
>
> _Example results:_
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/157e56ca-4dbe-406c-90ba-0418a08e3fb1)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/a16fa1f0-322b-4998-8f3d-ac0ccbb6a8b3)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/0c73baa1-123a-4f3f-a305-d7b8640cd90a)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/125ec8da-f033-4c7a-85b1-155578e90979)





### :white_check_mark: Model Evaluation
> ```
> |--------------------------------------------------|----------------------|---------------|------------|--------------|-------------|---------------|------------|--------------|-------------|----------|---------------------|------------------|--------------------|-------------------|------------------------|---------------------|-----------------------|
> |                      Model                       |   Sampling Method    | Precision (0) | Recall (0) | F1-Score (0) | Support (0) | Precision (1) | Recall (1) | F1-Score (1) | Support (1) | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score | Macro Avg Support | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1-Score |
> |--------------------------------------------------|----------------------|---------------|------------|--------------|-------------|---------------|------------|--------------|-------------|----------|---------------------|------------------|--------------------|-------------------|------------------------|---------------------|-----------------------|
> |               Logistic Regression                |     No Sampling      |     0.90      |    0.96    |     0.93     |   1171.00   |     0.69      |    0.45    |     0.54     |   237.00    |   0.87   |        0.79         |       0.70       |        0.74        |      1408.00      |          0.86          |        0.87         |         0.86          |
> |               Logistic Regression                |        SMOTE         |     0.94      |    0.74    |     0.83     |   1171.00   |     0.38      |    0.77    |     0.51     |   237.00    |   0.75   |        0.66         |       0.76       |        0.67        |      1408.00      |          0.85          |        0.75         |         0.78          |
> |               Logistic Regression                | Random Oversampling  |     0.95      |    0.76    |     0.84     |   1171.00   |     0.40      |    0.81    |     0.54     |   237.00    |   0.77   |        0.68         |       0.78       |        0.69        |      1408.00      |          0.86          |        0.77         |         0.79          |
> |               Logistic Regression                | Random Undersampling |     0.95      |    0.77    |     0.85     |   1171.00   |     0.42      |    0.81    |     0.56     |   237.00    |   0.78   |        0.69         |       0.79       |        0.70        |      1408.00      |          0.86          |        0.78         |         0.80          |
> |             Random Forest Classifier             |     No Sampling      |     0.97      |    0.99    |     0.98     |   1171.00   |     0.94      |    0.84    |     0.88     |   237.00    |   0.96   |        0.95         |       0.91       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |             Random Forest Classifier             |        SMOTE         |     0.97      |    0.97    |     0.97     |   1171.00   |     0.84      |    0.84    |     0.84     |   237.00    |   0.95   |        0.90         |       0.90       |        0.90        |      1408.00      |          0.95          |        0.95         |         0.95          |
> |             Random Forest Classifier             | Random Oversampling  |     0.98      |    0.98    |     0.98     |   1171.00   |     0.91      |    0.89    |     0.90     |   237.00    |   0.97   |        0.94         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |             Random Forest Classifier             | Random Undersampling |     0.98      |    0.89    |     0.93     |   1171.00   |     0.62      |    0.92    |     0.74     |   237.00    |   0.89   |        0.80         |       0.90       |        0.84        |      1408.00      |          0.92          |        0.89         |         0.90          |
> |          K Nearest Neighbors Classifier          |     No Sampling      |     0.89      |    0.95    |     0.92     |   1171.00   |     0.64      |    0.41    |     0.50     |   237.00    |   0.86   |        0.76         |       0.68       |        0.71        |      1408.00      |          0.85          |        0.86         |         0.85          |
> |          K Nearest Neighbors Classifier          |        SMOTE         |     0.97      |    0.81    |     0.88     |   1171.00   |     0.48      |    0.86    |     0.62     |   237.00    |   0.82   |        0.72         |       0.83       |        0.75        |      1408.00      |          0.88          |        0.82         |         0.84          |
> |          K Nearest Neighbors Classifier          | Random Oversampling  |     0.96      |    0.85    |     0.90     |   1171.00   |     0.53      |    0.82    |     0.64     |   237.00    |   0.85   |        0.74         |       0.84       |        0.77        |      1408.00      |          0.89          |        0.85         |         0.86          |
> |          K Nearest Neighbors Classifier          | Random Undersampling |     0.94      |    0.74    |     0.83     |   1171.00   |     0.37      |    0.76    |     0.50     |   237.00    |   0.74   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.74         |         0.77          |
> |          SMOTE and Logistic Regression           |     No Sampling      |     0.90      |    0.96    |     0.93     |   1171.00   |     0.69      |    0.45    |     0.54     |   237.00    |   0.87   |        0.79         |       0.70       |        0.74        |      1408.00      |          0.86          |        0.87         |         0.86          |
> |          SMOTE and Logistic Regression           |        SMOTE         |     0.94      |    0.74    |     0.83     |   1171.00   |     0.38      |    0.77    |     0.51     |   237.00    |   0.75   |        0.66         |       0.76       |        0.67        |      1408.00      |          0.85          |        0.75         |         0.78          |
> |          SMOTE and Logistic Regression           | Random Oversampling  |     0.95      |    0.76    |     0.84     |   1171.00   |     0.40      |    0.81    |     0.54     |   237.00    |   0.77   |        0.68         |       0.78       |        0.69        |      1408.00      |          0.86          |        0.77         |         0.79          |
> |          SMOTE and Logistic Regression           | Random Undersampling |     0.95      |    0.77    |     0.85     |   1171.00   |     0.42      |    0.81    |     0.56     |   237.00    |   0.78   |        0.69         |       0.79       |        0.70        |      1408.00      |          0.86          |        0.78         |         0.80          |
> |        SMOTE and Random Forest Classifier        |     No Sampling      |     0.97      |    0.99    |     0.98     |   1171.00   |     0.94      |    0.86    |     0.90     |   237.00    |   0.97   |        0.96         |       0.92       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |        SMOTE and Random Forest Classifier        |        SMOTE         |     0.97      |    0.97    |     0.97     |   1171.00   |     0.85      |    0.86    |     0.85     |   237.00    |   0.95   |        0.91         |       0.91       |        0.91        |      1408.00      |          0.95          |        0.95         |         0.95          |
> |        SMOTE and Random Forest Classifier        | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1171.00   |     0.93      |    0.87    |     0.90     |   237.00    |   0.97   |        0.95         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |        SMOTE and Random Forest Classifier        | Random Undersampling |     0.98      |    0.89    |     0.93     |   1171.00   |     0.63      |    0.91    |     0.75     |   237.00    |   0.90   |        0.81         |       0.90       |        0.84        |      1408.00      |          0.92          |        0.90         |         0.90          |
> |     SMOTE and K Nearest Neighbors Classifier     |     No Sampling      |     0.89      |    0.95    |     0.92     |   1171.00   |     0.64      |    0.41    |     0.50     |   237.00    |   0.86   |        0.76         |       0.68       |        0.71        |      1408.00      |          0.85          |        0.86         |         0.85          |
> |     SMOTE and K Nearest Neighbors Classifier     |        SMOTE         |     0.97      |    0.81    |     0.88     |   1171.00   |     0.48      |    0.86    |     0.62     |   237.00    |   0.82   |        0.72         |       0.83       |        0.75        |      1408.00      |          0.88          |        0.82         |         0.84          |
> |     SMOTE and K Nearest Neighbors Classifier     | Random Oversampling  |     0.96      |    0.85    |     0.90     |   1171.00   |     0.53      |    0.82    |     0.64     |   237.00    |   0.85   |        0.74         |       0.84       |        0.77        |      1408.00      |          0.89          |        0.85         |         0.86          |
> |     SMOTE and K Nearest Neighbors Classifier     | Random Undersampling |     0.94      |    0.74    |     0.83     |   1171.00   |     0.37      |    0.76    |     0.50     |   237.00    |   0.74   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.74         |         0.77          |
> |       Oversampling and Logistic Regression       |     No Sampling      |     0.90      |    0.96    |     0.93     |   1171.00   |     0.69      |    0.45    |     0.54     |   237.00    |   0.87   |        0.79         |       0.70       |        0.74        |      1408.00      |          0.86          |        0.87         |         0.86          |
> |       Oversampling and Logistic Regression       |        SMOTE         |     0.94      |    0.74    |     0.83     |   1171.00   |     0.38      |    0.77    |     0.51     |   237.00    |   0.75   |        0.66         |       0.76       |        0.67        |      1408.00      |          0.85          |        0.75         |         0.78          |
> |       Oversampling and Logistic Regression       | Random Oversampling  |     0.95      |    0.76    |     0.84     |   1171.00   |     0.40      |    0.81    |     0.54     |   237.00    |   0.77   |        0.68         |       0.78       |        0.69        |      1408.00      |          0.86          |        0.77         |         0.79          |
> |       Oversampling and Logistic Regression       | Random Undersampling |     0.95      |    0.77    |     0.85     |   1171.00   |     0.42      |    0.81    |     0.56     |   237.00    |   0.78   |        0.69         |       0.79       |        0.70        |      1408.00      |          0.86          |        0.78         |         0.80          |
> |    Oversampling and Random Forest Classifier     |     No Sampling      |     0.97      |    0.99    |     0.98     |   1171.00   |     0.94      |    0.85    |     0.89     |   237.00    |   0.97   |        0.95         |       0.92       |        0.94        |      1408.00      |          0.96          |        0.97         |         0.96          |
> |    Oversampling and Random Forest Classifier     |        SMOTE         |     0.97      |    0.97    |     0.97     |   1171.00   |     0.83      |    0.83    |     0.83     |   237.00    |   0.94   |        0.90         |       0.90       |        0.90        |      1408.00      |          0.94          |        0.94         |         0.94          |
> |    Oversampling and Random Forest Classifier     | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1171.00   |     0.94      |    0.87    |     0.90     |   237.00    |   0.97   |        0.96         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |    Oversampling and Random Forest Classifier     | Random Undersampling |     0.98      |    0.88    |     0.93     |   1171.00   |     0.62      |    0.92    |     0.74     |   237.00    |   0.89   |        0.80         |       0.90       |        0.83        |      1408.00      |          0.92          |        0.89         |         0.90          |
> | Oversampling and K Nearest Neighbors Classifier  |     No Sampling      |     0.89      |    0.95    |     0.92     |   1171.00   |     0.64      |    0.41    |     0.50     |   237.00    |   0.86   |        0.76         |       0.68       |        0.71        |      1408.00      |          0.85          |        0.86         |         0.85          |
> | Oversampling and K Nearest Neighbors Classifier  |        SMOTE         |     0.97      |    0.81    |     0.88     |   1171.00   |     0.48      |    0.86    |     0.62     |   237.00    |   0.82   |        0.72         |       0.83       |        0.75        |      1408.00      |          0.88          |        0.82         |         0.84          |
> | Oversampling and K Nearest Neighbors Classifier  | Random Oversampling  |     0.96      |    0.85    |     0.90     |   1171.00   |     0.53      |    0.82    |     0.64     |   237.00    |   0.85   |        0.74         |       0.84       |        0.77        |      1408.00      |          0.89          |        0.85         |         0.86          |
> | Oversampling and K Nearest Neighbors Classifier  | Random Undersampling |     0.94      |    0.74    |     0.83     |   1171.00   |     0.37      |    0.76    |     0.50     |   237.00    |   0.74   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.74         |         0.77          |
> |      Undersampling and Logistic Regression       |     No Sampling      |     0.90      |    0.96    |     0.93     |   1171.00   |     0.69      |    0.45    |     0.54     |   237.00    |   0.87   |        0.79         |       0.70       |        0.74        |      1408.00      |          0.86          |        0.87         |         0.86          |
> |      Undersampling and Logistic Regression       |        SMOTE         |     0.94      |    0.74    |     0.83     |   1171.00   |     0.38      |    0.77    |     0.51     |   237.00    |   0.75   |        0.66         |       0.76       |        0.67        |      1408.00      |          0.85          |        0.75         |         0.78          |
> |      Undersampling and Logistic Regression       | Random Oversampling  |     0.95      |    0.76    |     0.84     |   1171.00   |     0.40      |    0.81    |     0.54     |   237.00    |   0.77   |        0.68         |       0.78       |        0.69        |      1408.00      |          0.86          |        0.77         |         0.79          |
> |      Undersampling and Logistic Regression       | Random Undersampling |     0.95      |    0.77    |     0.85     |   1171.00   |     0.42      |    0.81    |     0.56     |   237.00    |   0.78   |        0.69         |       0.79       |        0.70        |      1408.00      |          0.86          |        0.78         |         0.80          |
> |    Undersampling and Random Forest Classifier    |     No Sampling      |     0.97      |    0.99    |     0.98     |   1171.00   |     0.94      |    0.85    |     0.89     |   237.00    |   0.97   |        0.96         |       0.92       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |    Undersampling and Random Forest Classifier    |        SMOTE         |     0.97      |    0.97    |     0.97     |   1171.00   |     0.85      |    0.83    |     0.84     |   237.00    |   0.95   |        0.91         |       0.90       |        0.90        |      1408.00      |          0.95          |        0.95         |         0.95          |
> |    Undersampling and Random Forest Classifier    | Random Oversampling  |     0.97      |    0.99    |     0.98     |   1171.00   |     0.94      |    0.86    |     0.90     |   237.00    |   0.97   |        0.95         |       0.93       |        0.94        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |    Undersampling and Random Forest Classifier    | Random Undersampling |     0.98      |    0.90    |     0.94     |   1171.00   |     0.64      |    0.92    |     0.76     |   237.00    |   0.90   |        0.81         |       0.91       |        0.85        |      1408.00      |          0.93          |        0.90         |         0.91          |
> | Undersampling and K Nearest Neighbors Classifier |     No Sampling      |     0.89      |    0.95    |     0.92     |   1171.00   |     0.64      |    0.41    |     0.50     |   237.00    |   0.86   |        0.76         |       0.68       |        0.71        |      1408.00      |          0.85          |        0.86         |         0.85          |
> | Undersampling and K Nearest Neighbors Classifier |        SMOTE         |     0.97      |    0.81    |     0.88     |   1171.00   |     0.48      |    0.86    |     0.62     |   237.00    |   0.82   |        0.72         |       0.83       |        0.75        |      1408.00      |          0.88          |        0.82         |         0.84          |
> | Undersampling and K Nearest Neighbors Classifier | Random Oversampling  |     0.96      |    0.85    |     0.90     |   1171.00   |     0.53      |    0.82    |     0.64     |   237.00    |   0.85   |        0.74         |       0.84       |        0.77        |      1408.00      |          0.89          |        0.85         |         0.86          |
> | Undersampling and K Nearest Neighbors Classifier | Random Undersampling |     0.94      |    0.74    |     0.83     |   1171.00   |     0.37      |    0.76    |     0.50     |   237.00    |   0.74   |        0.65         |       0.75       |        0.66        |      1408.00      |          0.84          |        0.74         |         0.77          |
> |                XGBoost Classifier                |     No Sampling      |     0.97      |    0.98    |     0.98     |   1171.00   |     0.92      |    0.85    |     0.88     |   237.00    |   0.96   |        0.94         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |                XGBoost Classifier                |        SMOTE         |     0.97      |    0.98    |     0.97     |   1171.00   |     0.89      |    0.86    |     0.87     |   237.00    |   0.96   |        0.93         |       0.92       |        0.92        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |                XGBoost Classifier                | Random Oversampling  |     0.98      |    0.98    |     0.98     |   1171.00   |     0.92      |    0.91    |     0.92     |   237.00    |   0.97   |        0.95         |       0.95       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |                XGBoost Classifier                | Random Undersampling |     0.99      |    0.90    |     0.94     |   1171.00   |     0.66      |    0.94    |     0.77     |   237.00    |   0.91   |        0.82         |       0.92       |        0.86        |      1408.00      |          0.93          |        0.91         |         0.91          |
> |           Oversamp XGBoost Classifier            |     No Sampling      |     0.97      |    0.98    |     0.98     |   1171.00   |     0.92      |    0.85    |     0.88     |   237.00    |   0.96   |        0.94         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |           Oversamp XGBoost Classifier            |        SMOTE         |     0.97      |    0.98    |     0.97     |   1171.00   |     0.89      |    0.86    |     0.87     |   237.00    |   0.96   |        0.93         |       0.92       |        0.92        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |           Oversamp XGBoost Classifier            | Random Oversampling  |     0.98      |    0.98    |     0.98     |   1171.00   |     0.92      |    0.91    |     0.92     |   237.00    |   0.97   |        0.95         |       0.95       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |           Oversamp XGBoost Classifier            | Random Undersampling |     0.99      |    0.90    |     0.94     |   1171.00   |     0.66      |    0.94    |     0.77     |   237.00    |   0.91   |        0.82         |       0.92       |        0.86        |      1408.00      |          0.93          |        0.91         |         0.91          |
> |         Undersampled XGBoost Classifier          |     No Sampling      |     0.97      |    0.98    |     0.98     |   1171.00   |     0.92      |    0.85    |     0.88     |   237.00    |   0.96   |        0.94         |       0.92       |        0.93        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |         Undersampled XGBoost Classifier          |        SMOTE         |     0.97      |    0.98    |     0.97     |   1171.00   |     0.89      |    0.86    |     0.87     |   237.00    |   0.96   |        0.93         |       0.92       |        0.92        |      1408.00      |          0.96          |        0.96         |         0.96          |
> |         Undersampled XGBoost Classifier          | Random Oversampling  |     0.98      |    0.98    |     0.98     |   1171.00   |     0.92      |    0.91    |     0.92     |   237.00    |   0.97   |        0.95         |       0.95       |        0.95        |      1408.00      |          0.97          |        0.97         |         0.97          |
> |         Undersampled XGBoost Classifier          | Random Undersampling |     0.99      |    0.90    |     0.94     |   1171.00   |     0.66      |    0.94    |     0.77     |   237.00    |   0.91   |        0.82         |       0.92       |        0.86        |      1408.00      |          0.93          |        0.91         |         0.91          |
> |--------------------------------------------------|----------------------|---------------|------------|--------------|-------------|---------------|------------|--------------|-------------|----------|---------------------|------------------|--------------------|-------------------|------------------------|---------------------|-----------------------|
> ```


## 	:triangular_flag_on_post:  Key Benefits and Further Analysis
หลังจากที่มีการ Run model ตาม Algorithm ต่าง ๆ และแบ่งการทดสอบชุดข้อมูลเป็น train และ test เพื่อทดสอบความแม่นยำของ model เมื่อเราได้ model ที่มีความแม่นยำแล้ว จะทำให้ธุรกิจสามารถประเมินได้ว่าลูกค้าลักษณะรายใดบ้างที่มีโอกาสหายไปจากธุรกิจ และตัวแปรใดที่มีผลต่อธุรกิจนั้น นอกจากนี้ หากเราสามารถวิเคราะห์ Mean time between purchase และ Cohort Analysis ควบคู่เพื่อดูว่าช่วงเวลาใดบ้างที่เราควรจะเข้าถึงลูกค้า เช่น มีการส่ง promotion หรือ ทำ loyalty program ต่าง ๆ ก่อนที่การ churn จะเกิดขึ้น รวมถึงการทำ Voice of Customer ผ่านการทำ NLP จะช่วยให้ธุรกิจสามารถทราบข้อดี ข้อเสีย หรือข้อควรปรับปรุงซึ่งเป็นความเห็นลูกค้า และสามารถนำมาปรับใช้ธุรกิจเพื่อตอบสนองต่อความต้องการของลูกค้าได้ดียิ่งขึ้น

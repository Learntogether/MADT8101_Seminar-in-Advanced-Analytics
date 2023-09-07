# Workshop#02_Customer Single View and Basic Customer Analysis

## Customer Single View คืออะไร และสำคัญอย่างไร?
**Customer Single View** คือ การสรุปข้อมูลทุกอย่างเกี่ยวกับลูกค้า ซึ่งจำเป็นแก่การนำไปใช้งานหรือวิเคราะห์ข้อมูล โดยเป็นการจัด Format ข้อมูลของลูกค้า 1 ราย ให้อยู่ภายใน 1 record

**_แล้วทำไมถึงต้องสรุปข้อมูลให้อยู่ภายใน 1 record...?_**

ลองนึกภาพธุรกิจ Supermarket ซึ่งมีลูกค้าเข้ามาใช้บริการในแต่ละวันเป็นจำนวนมาก และลูกค้าบางใน 1 วัน อาจมีการเข้ามาซื้อสินค้ามากกว่า 1 ครั้ง หรือในแต่ละเดือนเข้ามาซื้อสินค้า 15 ครั้ง และแต่ละครั้งมียอดซื้อสินค้า และประเภทสินค้าที่ซื้อไม่ซ้ำกัน ซึ่งหากไม่มีการสรุปข้อมูลของลูกค้ารายนั้นเก็บไว้ ก็จะไม่สามารถนำข้อมูลมาวิเคราะห์ในภาพรวมได้ ดังนั้น จึงต้องมีการนำข้อมูล Transaction ของลูกค้าแต่ละรายมาสรุปให้อยู่ใน 1 record เพื่อให้สามารถนำข้อมูลเหล่านั้นไปวิเคราะห์ข้อมูล เพื่อให้สามารถกำหนดกลยุทธ์ที่สามารถดึงดูด และตอบโจทย์ความต้องการของลูกค้าได้ตามตัวอย่างด้านล่าง
![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/26655cfd-f205-405d-ad90-154a0f7c5224)

**ตัวอย่างการสรุป Transaction data ให้อยู่ในรูป Customer Single View**

![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/24832bfe-835a-4101-b5b7-e57486fcd09b)
จากภาพด้านบนเป็นตัวอย่าง Transaction ของธุรกิจ Supermarket ประกอบด้วยลูกค้าที่เป็น Membership และ Non-Membership ซึ่งลูกค้าโดยส่วนใหญ่จะมี Transaction หลายรายการ และแต่ละรอบการซื้อมี Amount แตกต่างกัน ดังนั้น จึงได้มีการนำข้อมูล Transaction ของลูกค้าแต่ละรายมาสรุปให้อยู่ใน 1 record ตามตัวอย่างในรูปภาพด้านล่าง

![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/79392394-237a-4762-b994-2ab2cfdf782c)

ซึ่งมีการกำหนด Features ที่สำคัญที่มีประโยชน์ในการนำไปวิเคราะห์ดังนี้
* Total Spending = Total spending by Customer id
* Purchased time = Count of Basket_Id. (Frequency)
* Average spending per basket ID = Total spending / Purchased time
* First time purshase date = First date that customer perchased
* Lastest purchase date = Last date that customer perchased
* Member since = Last date on Dataset (31/07/2008) - First time purshase date
* Lifetime = First time purshase date - Lastest purchase date
* Mean time between perchase = Lifetime/Purchased time
* Customer Lifetime Value = Average spending per basket ID * Lifetime (Day)



## Basic Customer Analysis
จาก Customer Single View จะสามารถเอาข้อมูลข้างต้นมาวิเคราะห์ Customer insight เพื่อทำ Customer segmentation และกำหนดกลยุทธ์ทางการตลาดกับลูกค้าได้เบื้องต้น ตามตัวอย่างดังนี้

### Analysis Objective
วิเคราะห์ Customer insight เพื่อทำ Customer segmentation เพื่อให้สามารถกำหนดกลยุทธ์ทางการตลาดที่เหมาะสมกับลูกค้าแต่ละกลุ่ม
  * ทำ Visualized เพื่อวิเคราะห์ข้อมูล Dataset เบื้องต้น 
![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/2ec08338-2885-4f9e-a6ce-e98f0ca84e5e)

### Customer Segmentation
  * นำข้อมูลเข้า Dataiku และใช้ Auto ML - Clustering และ K-Mean algorithm เพื่อทำ Customer Segmentation
      *  ค่า Silhouette สูงสุดจะอยู่ที่ K = 5
![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/f9f66790-d012-416b-bd07-a1529dbf93fa)
      *  Scatter plot - Average spending per Basket ID vs. Mean time between purchase
        ![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/c02cf02c-1610-4c86-b2da-cc5c9915e744)
      *  Scatter plot - Customer Lifetime Value vs. Mean time between purchase
![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/7f34b523-451c-4b29-9621-2a21a5a72862)

### Result and Actions
จากผลการทำ Clustering โดยใช้ K-Mean จะแบ่งลูกค้าออกเป็น 5 กลุ่ม ซึ่งแต่ละกลุ่มจะมี Character และ Key actions ที่แตกต่างกัน ตามรายละเอียดด้านล่าง
![image](https://github.com/Learntogether/Workshop02_Customer-Single-View/assets/136689632/f7a11d51-f3fe-4711-813c-bbd4feee5578)


### Further Analysis and Next best actions
  * Customer Behavior and Insight Analysis - วิเคราะห์ปัจจัยที่ส่งผลกระทบต่อการตัดสินใจซื้อของลูกค้าแต่ละกลุ่ม เช่น ลูกค้าบางกลุ่มจะ sensitive กับส่วนลด โปรแกรมสะสมแต้มเป็น
  * Sentiment Analysis - วิเคราะห์ความต้องการของลูกค้า และความรู้สึกของลูกค้าที่มีต่อธุรกิจ เพื่อนำมาพัฒนา และปรับปรุงให้ดีขึ้น และรักษาฐานลูกค้าให้อยู่กับธุรกิจ
  * Social/Trend Analysis - วิเคราะห์ความต้องการ สถานการณ์ ณ ปัจจุบัน สินค้าที่กำลังเป็นกระแส เพื่อให้มีสินค้าพร้อมขาย และสต๊อคสินค้าอย่างมีประสิทธิภาพ เพียงพอต่อความต้องการของลูกค้า


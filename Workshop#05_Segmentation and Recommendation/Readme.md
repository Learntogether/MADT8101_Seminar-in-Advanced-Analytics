# Workshop#05_Segmentation and Recommendation

## 	:point_right: Customer Segmentation คืออะไร?
**_Customer Segmentation_** คือ การวิเคราะห์แบ่งกลุ่มลูกค้าออกเป็นกลุ่มย่อย ๆ โดยอ้างอิงข้อมูลของลูกค้าที่มีลักษณะคล้าย ๆ ตั้งแต่ข้อมูลลักษณะทางคุณภาพ และเชิงปริมาณต่าง ๆ พวก Demographic ตลอดจนถึงพฤติกรรมการบริโภคสินค้า/บริการ
ดังนั้น Customer Segmentation จึงเป็นเครื่องมือสำคัญที่นักการตลาดใช้ในการระบุความต้องการของกลุ่มลูกค้า เพื่อที่ธุรกิจจะสามารถชนะใจลูกค้า ชนะคู่แข่ง ด้วยการพัฒนาผลิตภัณฑ์/บริการ และออกแคมเปญ หรือโปรโมชั่นต่าง ๆ ที่ตอบสนองลูกค้าในแต่ละกลุ่ม
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/b171b7dc-2145-409e-8476-e40a0164360f)
_ภาพจาก: https://commons.wikimedia.org/wiki/File:Customer_Segmentation.png_


## 	:point_right: Machine Learning - Clustering สำหรับแบ่งกลุ่มลูกค้า
Clustering Algorithms มีหลากหลายประเภท ซึ่งจะยกตัวอย่าง 4 ประเภทหลัก ๆ ดังนี้ 

_ข้อมูลจาก: https://bigdata.go.th/big-data-101/4-types-of-clustering/_
> :green_circle: **(1) Centroid-based Clustering**  เป็นการแบ่งกลุ่ม cluster แบบ centroid-based
> การแบ่งกลุ่มของข้อมูลเกิดจากจุดข้อมูล (Data Point) ที่กระจุกตัวใกล้กับจุดกึ่งกลางของกลุ่ม (Centroid) มากที่สุด การแบ่งกลุ่มแบบนี้มีประสิทธิภาพแต่ก็มีความอ่อนไหวขึ้นกับเงื่อนไขตั้งต้นและข้อมูลที่เป็น outlier ซึ่ง algorithm ที่นิยมใช้และถูกกล่าวถึงบ่อยครั้ง คือ k-means เพราะเป็น algorithm ที่ง่ายและมีประสิทธิภาพ โดยมีการทำงานแบบ iterative คือวนซ้ำ ๆ เพื่อคำนวณระยะห่าง หรือ Euclidean Distance ของ data point กับ centroid ในแต่ละกลุ่ม ผลที่ได้คือ data point ที่อยู่ใกล้กับ centroid ไหนมากที่สุด ก็จะถูกจัดให้อยู่กลุ่มเดียวกันกับ centroid นั้นมีข้อควรระลึกไว้คือ ผลลัพธ์จากการใช้ k-means ในแต่ละครั้งจะไม่เหมือนกันเนื่องจากการกำหนด centroid ในขั้นแรกสุดนั้นเป็นการสุ่ม
> 
> _ตัวอย่างการใช้งาน_: การแบ่ง segment ของลูกค้าในประเภทธุรกิจต่าง ๆ การแบ่งกลุ่มเอกสารที่ไม่มีประเภทระบุ และการแนะนำสินค้าที่ลูกค้ามีโอกาสจะซื้อ
> 
> :green_circle: **(2) Density-based Clustering** เป็นการแบ่งกลุ่ม cluster แบบ density-based
> การแบ่งกลุ่มของข้อมูลเกิดจากการกระจุกตัวของ data point ที่เกาะกันอย่างหนาแน่นและไม่เป็นรูปลักษณ์ที่ตายตัว อุปสรรคของการแบ่งกลุ่มแบบนี้ คือ ความหนาแน่นที่ผันผวนและจำนวน features ของชุดข้อมูล นอกจากนี้ยังแยก outlier ออกจากกลุ่มได้ชัดเจนด้วย ถือว่าข้อดีและเป็นการแก้ข้อบกพร่องของ centroid-based model ที่ sensitive กับ > outlier ซึ่ง algorithm ที่นิยมใช้ คือ DBSCAN (Density-Based Spatial Clustering of Applications with Noise) ในขั้นแรก กำหนดรัศมีจากจุดศูนย์กลาง (eps) และจำนวน data point ขั้นต่ำในรัศมี (MinPts) จากนั้นถ้า data point ที่เป็นจุดศูนย์กลางรวมกับ data point ที่อยู่โดยรอบภายในวงรัศมีมีจำนวนเท่ากับ MinPts เราจะเรียก data point จุดนั้นว่า “core point” ส่วน “border” คือ data point ที่เป็นจุดศูนย์กลางและมี data point ที่อยู่โดยรอบกับ core point หรือ border ด้วยกันเอง พูดง่าย ๆ คือจะรวมจุดที่เป็นเพื่อนของเพื่อนของเพื่อนไปเรื่อย ๆ จนไม่มีเพื่อนให้จับกลุ่มอีกแล้ว ก็จะถือว่าสิ้นสุดการจับกลุ่ม ส่วนจุดที่ไม่ถูกจับรวมกลุ่มเพราะอยู่ไกลเกินไปจะถือว่าเป็น outlier หรือ noise
> 
> _ตัวอย่างการใช้งาน_: Anomaly Detection ที่มีมากกว่า 1 ตัวแปร
> 
> :green_circle: **(3) Distribution-based Clustering** เป็นการแบ่งกลุ่ม cluster แบบ distribution-based
> การแบ่งกลุ่มประเภทนี้สันนิษฐานว่าข้อมูลมีรูปแบบการแจกแจงแบบใดแบบหนึ่ง เช่น การแจกแจงปกติ (Normal Distributions) เมื่อระยะห่างระหว่างจุดศูนย์กลางของการแจกแจง กับ data point เพิ่มมากขึ้น ความน่าจะเป็นที่ data point เป็นส่วนหนึ่งของการแจกแจงนั้นจะลดลง แต่ถ้าเราไม่ทราบว่าข้อมูลมีการแจกแจงแบบใด ก็ควรเลือกใช้การ clustering รูปแบบอื่น ซึ่ง algorithm ที่เป็นตัวอย่าง คือ Expectation-maximization algorithm หรือเรียกว่า EM algorithm ซึ่งมีการทำงานแบบ iterative ระหว่าง 2 โหมด คือ E-Step ประมาณการค่าของตัวแปรที่หายไปจากชุดข้อมูล และ M-Step เพิ่มประสิทธิภาพพารามิเตอร์ของโมเดล ตัวอย่างการใช้งาน เช่น การตรวจสอบหาการทุจริต (Fraud Detection) 
> 
> _ตัวอย่างการใช้งาน_: การตรวจสอบหาการทุจริต (Fraud Detection) 
> 
> :green_circle: **(4) Hierarchical Clustering** เป็นการแบ่งกลุ่ม cluster แบบ hierarchical ของ plant kingdom
> การแบ่งกลุ่มประเภทนี้จะสร้างให้เกิดต้นไม้ของกลุ่มข้อมูลขึ้น เหมาะสำหรับข้อมูลที่มีลำดับชั้น เช่น อนุกรมวิธาน (Taxonomy) การแบ่งกลุ่มลักษณะนี้มี 2 ประเภท คือ ล่างขึ้นบน (Agglomerative) และ บนลงล่าง (Divisive) ดังนี้ 
> * Agglomerative – ในเริ่มแรก data point นั้นนับเป็นหนึ่ง cluster จากนั้นจะคำนวณหาค่าความใกล้ชิด cluster ที่อยู่ใกล้กันจะถูกจับรวมตัวกัน และจะวนทำเช่นนี้ไปเรื่อย ๆ จนกว่าจะกลายเป็น cluster เดียวในที่สุด แผนภาพที่ถูกใช้นำเสนอการทำ cluster เช่นนี้ คือ Dendrogram  
> * Divisive – เทคนิคนี้จะทำตรงกันข้ามกับ Agglomerative คือเริ่มจาก cluster กลุ่มใหญ่กลุ่มเดียว และแยกกลุ่มที่ไม่เหมือนกันออกไปเรื่อย ๆ จะเป็นเป็น n กลุ่มที่แยกต่อไม่ได้แล้ว 
> 
> _ตัวอย่างการใช้งาน_: การแบ่งกลุ่มพืช หรือสิ่งมีชีวิต, การแบ่งกลุ่มสินค้า 
> 
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/edad11f0-fbdb-4215-85c3-0f0a937fcd0e)


## 	:point_right: Workshop - Segmentation and Recommendation
### :white_check_mark: Python code
> * Cleasing json file format
> * Preparing variables (Ent_Downline and Sponsor_Downline)
> * [Product Recommendation](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2305_Segmentation%20and%20Recommendation/Recommendation_model_final.ipynb)

### :white_check_mark: Understand business
> **Business:** A a Social Network Marketing organisation
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/9b87c2ad-8173-438f-81ea-8af5e126fb88)

### :white_check_mark: Define Analysis Objective
> **Analysis Objective:** Create a personalized marketing through customer segmentation and product recommendation system.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/68bf89a5-ff2f-481b-a2a1-d28a2bf48750)

### :white_check_mark: Data Pre-Processing
> Determined to drop some users who have no transaction, also users who joined before the Company founded.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/2421645f-2b51-48a4-94c9-32398da047fc)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1b043da8-e841-4038-9d79-0197ccb6f108)

### :white_check_mark: Exploratory Data Analysis (EDA) and Preparation by DATAIKU
> :green_circle: **Sanity check:** Understand the structure of dataset, then check outlier and missing value.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/7e7deddb-adfd-4552-aeaa-386cb5c10fde)
>
> :green_circle: **Features Preparation:** Preparing the features for train the model, then generate customer single view and EDA the correlation and PCA of variables.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1dec1e5c-e678-48c8-861a-8916b255e971)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/96d4960a-dbf6-4970-b993-b9811273812d)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1bcb7f94-2949-4f94-8ab9-75e52a86579c)


### :white_check_mark: Segmentation and Actions
> :green_circle: **Segmentation:** Segmentation by Auto-ML Clustering module of DATAIKU with the K-Mean algorithm. I decided to split customers into 4 segments based on silhouette score, Elbow method and the Customer's character of each segments.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/0ecbc949-3420-4c40-bbc3-5e7c4c76feb5)
> :green_circle: The features that I selected to train the model, the right-side graph shows the percentage important of each variable that effect to the model.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/358c4e2d-7919-4d5f-b465-4ec90c57adc6)
>
> :green_circle: **Intepret results and Actions**
> 
> 

### :white_check_mark: Recommendation
> :green_circle: **Product recommendation** by using Content based and Collaborative filtering.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/d2928b7b-a50e-4a15-8dc8-f689b18bc032)
>
> :green_circle: **Data for Recommendation System** Using DATAIKU for Group the data by ent, product_cdoe and sum of Product_Qty, then mapping the cluster_labels from segmentation process.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/73a7d8d1-0a36-4350-bb60-2a92496800c3)
>
> :green_circle: **Libraries:** pandas, cosine_similarity and csr_matrix.
> ```
> # Import necessary libraries
> import pandas as pd
> from sklearn.metrics.pairwise import cosine_similarity
> from scipy.sparse import csr_matrix
> ```
> 
> :green_circle: **Create the matrix:** User to Item, Item to Item and User to User.
> 
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/77238ca1-86a6-4f14-a80b-000790c5231c)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/82b7992f-c795-4181-80a8-4bf874243fd0)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/4081e112-80bf-4517-8468-1540555e93db)
>
> :green_circle: **Recommend items:** Recommend items based on the similarity items that others users buy with this product.
> Select the 0C1CC1, then recommend top 10 product with highest similarity score.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/47f91afc-a97d-463b-ac92-586c04df38e5)
> 
> :green_circle: **Recommend items for user:** Recommend item of each user by the similarity of others user basket.
> Random user for recommend the products.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/68b4216f-2167-4b1d-ad0c-5f2da797f040)
> 

## 	:point_right: Further Analysis for next best actions
* Customer Segmentation Movement
* 

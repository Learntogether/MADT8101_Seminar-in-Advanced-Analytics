# Workshop#05_Segmentation and Recommendation

## 	:point_right: Customer Segmentation คืออะไร?
**_Customer Segmentation_** คือ การวิเคราะห์แบ่งกลุ่มลูกค้าออกเป็นกลุ่มย่อย ๆ โดยอ้างอิงข้อมูลของลูกค้าที่มีลักษณะคล้าย ๆ ตั้งแต่ข้อมูลลักษณะทางคุณภาพ และเชิงปริมาณต่าง ๆ พวก Demographic ตลอดจนถึงพฤติกรรมการบริโภคสินค้า/บริการ
ดังนั้น Customer Segmentation จึงเป็นเครื่องมือสำคัญที่นักการตลาดใช้ในการระบุความต้องการของกลุ่มลูกค้า เพื่อที่ธุรกิจจะสามารถชนะใจลูกค้า ชนะคู่แข่ง ด้วยการพัฒนาผลิตภัณฑ์/บริการ และออกแคมเปญ หรือโปรโมชั่นต่าง ๆ ที่ตอบสนองลูกค้าในแต่ละกลุ่ม

![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/db799395-4a92-4341-baed-4863b0fdeb09)

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
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/dc47ce28-ecd4-4357-8e50-9995cfe33cf5)


## 	:point_right: Workshop - Segmentation and Recommendation
### :white_check_mark: Python code
> * [Preparing variables (Ent_Downline and Sponsor_Downline)](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2305_Segmentation%20and%20Recommendation/Preparing%20variables.ipynb)
> * [Product Recommendation](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2305_Segmentation%20and%20Recommendation/Recommendation_model_final.ipynb)

### :white_check_mark: Understand business
> **Business:** A a Social Network Marketing organisation
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/fc3559c3-32de-41d2-8620-8ce4f5fc740b)

### :white_check_mark: Define Analysis Objective
> **Analysis Objective:** Create a personalized marketing through customer segmentation and product recommendation system.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/537876ba-f18b-4912-ab50-6551650b0a1b)

### :white_check_mark: Data Pre-Processing
> Determined to drop some users who have no transaction, also dropped users who joined before the Company founded.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/653d5062-a38f-4048-9588-ef95677a4d28)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/add45cad-f747-4250-90b8-d8897ee9874d)


### :white_check_mark: Exploratory Data Analysis (EDA) and Preparation by DATAIKU
> :green_circle: **Sanity check:** Understand the structure of dataset, then check outlier and missing value.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/5da95327-3506-42e2-af8d-c9ba03754495)
>
> :green_circle: **Features Preparation:** Preparing the features for train the model, then generate customer single view and EDA the correlation and PCA of variables.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/a456310c-3a12-415f-b402-64088c1a351a)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/a5ec138e-4919-45e5-88fd-01cff9c7fcbc)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/52fdb8fa-b349-4775-a2e1-643f35cd262f)


### :white_check_mark: Segmentation and Actions
> :green_circle: **Segmentation:** Segmentation by Auto-ML Clustering module of DATAIKU with the K-Mean algorithm. I decided to split customers into 4 segments based on silhouette score, Elbow method and the Customer's character of each segments.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/46d1bbb4-0c7a-4309-9d49-ff06d950c66e)
> :green_circle: The features that I selected to train the model, the right-side graph shows the percentage important of each variable that effect to the model.
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/c572f6c4-bff7-4ad6-b1ff-0381dbabd37e)
>
> :green_circle: **Interpreting results and Actions**
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/055e8523-1f3c-4901-ae98-d133bba3fb80)

> 

### :white_check_mark: Recommendation
> :green_circle: **Product recommendation** by using Content based and Collaborative filtering.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1e7b98f0-94e2-42d9-a0cb-3c7a562fdd54)
>
> :green_circle: **Data for Recommendation System** Using DATAIKU for Group the data by ent, product_cdoe and sum of Product_Qty, then mapping the cluster_labels from segmentation process.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/499157cd-7338-4982-870a-be8630fee63b)
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
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/4af49a85-ca67-45a3-811f-502316329523)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/3596f85b-4c20-4d23-ab29-9f79df91c572)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/67b4674b-6d00-40fb-9674-2c8d8f09617e)
>
> :green_circle: **Recommend items:** Recommend items based on the similarity items that others users buy with this product.
> Select the 0C1CC1, then recommend top 10 product with highest similarity score.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/828d045a-3888-4e66-9670-2805fcc7c62a)
> 
> :green_circle: **Recommend items for user:** Recommend item of each user by the similarity of others user basket.
> Random user for recommend the products.
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/2f6f7263-3be6-4695-bce6-869e22c25567)
> 

## 	:triangular_flag_on_post: Further analysis for next best actions
* **Customer Movement Analysis:** เพื่อวิเคราะห์พฤติกรรมของลูกค้า ณ ช่วงเวลาใด เวลาหนึ่ง ทำให้ทราบพฤติกรรมที่เปลี่ยนแปลงไป และปรับกลยุทธ์ทางการตลาดให้สอดคล้องกับลูกค้า หรือกระตุ้นลูกค้าเพิ่มขึ้น
* **Voice of Customers / Sentiment Analysis:** เพื่อวิเคราะห์มุมมองของลูกค้าที่มีต่อบริษัท feedback ต่าง ๆ เปรียบเทียบกับคู่แข่งในอุตสาหกรรม และเพื่อพัฒนาผลิตภัณฑ์/บริการ/กระบวนการในอนาคต

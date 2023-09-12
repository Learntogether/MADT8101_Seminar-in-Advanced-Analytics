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
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/d045bd62-309f-450b-b91b-8038a9bc677d)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/cac4b55b-75f0-480c-bbc8-0851b95b3fb6)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/620901b1-e8d0-4ff2-8d79-a29379385e66)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/9b87c2ad-8173-438f-81ea-8af5e126fb88)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/b11bfa3d-ffbd-47d3-8731-a75516ca85af)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/68bf89a5-ff2f-481b-a2a1-d28a2bf48750)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/98f99177-b5f7-4a6e-992b-f1473bebc456)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/2421645f-2b51-48a4-94c9-32398da047fc)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1b043da8-e841-4038-9d79-0197ccb6f108)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/22b0d100-0a58-4fab-a61b-baa74b9c39d4)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/7e7deddb-adfd-4552-aeaa-386cb5c10fde)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1f4fd621-e7cd-457b-a1d5-2ca61f76f717)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/1dec1e5c-e678-48c8-861a-8916b255e971)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/7cb2ef88-1924-489a-9333-14f5037e74bc)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/216a55e2-dcc8-4a5f-8022-896914a04742)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/90ea1a4e-3523-4d9a-932d-170c79b79f1e)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/0ecbc949-3420-4c40-bbc3-5e7c4c76feb5)
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/358c4e2d-7919-4d5f-b465-4ec90c57adc6)






## 	:point_right: Further Analysis
* Customer Segmentation Movement
* 

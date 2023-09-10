# Workshop#06_Voice of Customer

## 	:point_right: Voice of Customer คืออะไร?
**Voice of Customer (VOC) หรือ เสียงจากลูกค้า** คือ การรับฟังข้อมูล ความเห็น การติชม ข้อเสนอแนะของลูกค้าที่มีต่อผลิตภัณฑ์/บริการของบริษัท รวมถึงกระบวนการต่าง ๆ ที่เกี่ยวข้องกับการให้บริการลูกค้า และธุรกิจ หรือความเห็นที่มีความสำคัญกับการพัฒนาผลิตภัณฑ์/บริการของบริษัท ซึ่งปัจจุบัน โดยรวมแบ่งออกเป็น 3 อย่าง ได้แก่
* Contact VOC คือ เสียงหรือความเห็นของลูกค้าที่ได้รับจากการสนทนาโต้ตอบระหว่างบริษัทกับลูกค้า ไม่ว่าจะเป็นช่องทางโทรศัพท์ อีเมล หรือแชทต่าง ๆ
* Social VOC คือ เสียงหรือความเห็นของลูกค้าที่ปรากฎอยู่ใน Social Media Platform ต่าง ๆ เช่น Facebook, Twitter รวมถึงหากบริษัทมีการขายผลิตภัณฑ์/บริการผ่านช่องทางออนไลน์ (e-commerce) ต่าง ๆ
* Survey VOC คือ เสียงหรือความเห็นของลูกค้าที่บริษัทได้จากแบบสอบถาม (Survey) ที่จัดทำขึ้นเพื่อเก็บข้อมูลความเห็นของลูกค้าโดยตรง
  
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/dcca7eaa-9412-4467-a8ab-795a0e905d4d)

_ภาพจาก: https://www.talkwalker.com/blog/voice-of-the-customer_



## 	:point_right: Voice of Customer analytics tools
* Sentiment Analysis
* Fake review identification 
* Topic identification
* Net promoter score
* Named entity recognition


## 	:point_right: Topic Modeling with LDA
**Latent Dirichlet Allocation** คือ Latent Probabilistic Model ที่ใช้ในการทำ Topic Modelling โดย Topic ที่กล่าวถึงนี้อาจจะเป็น หัวข้อ หรือ theme อะไรบางอย่างที่ซ่อน (hidden) อยู่ในเอกสารและเป็นสิ่งที่เราต้องการ extract มาจากเอกสาร แต่เราและคอมพิวเตอร์ไม่เห็นมันตรง ๆ หมายความ คือ เราจะสามารถบอกได้แค่ว่า คำเหล่านี้อยู่ใน latent topic ที่ 1 2 หรือ 3 และเอกสารนี้มีสัดส่วนของแต่ละ latent topic เท่าไร ซึ่งจะต้องมากำหนดว่าแต่ละ latent topic นั้นกล่าวถึงอะไร เช่น latent topic ที่ 1 กล่าวถึงเรื่องการเมือง latent topic ที่ 2 กล่าวถึงเรื่องกีฬา รวมถึงสามารถระบุได้ว่า topic 1 และ มีคำต่าง ๆ กระจายด้วยความน่าจะเป็นที่เท่าไหร่ ซึ่งเราใช้ความน่าจะเป็นในการอธิบาย topic ต่าง ๆ ที่กระจายอยู่ 

นอกจากนี้ LDA ยังถือเป็น Generative Model หมายความว่า หลังจากที่เราเรียนรู้ความน่าจะเป็นต่าง ๆ จากเอกสาร และคำ ที่เรามีแล้ว การจะสร้างเอกสารใหม่ก็ไม่ใช่เรื่องยากอะไร ก็แค่สุ่ม topic และสุ่มคำจาก topic นั้นมาสร้างเอกสารใหม่ สำหรับวิธีการ generate เอกสารใหม่นั้น เค้าก็เรียกกันกลาง ๆ กันว่า Generative Process

 ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/a71be106-334f-43a7-871b-95965b4f6940)

 _แหล่งที่มาข้อมูล: https://pongsakorn-jrc.medium.com/%E0%B8%A3%E0%B8%B9%E0%B9%89%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%81%E0%B8%B1%E0%B8%9A-latent-dirichlet-allocation-part-1-2495acfcda86_

## 	:point_right: Workshop - Topic Modeling with LDA
### :white_check_mark: Dataset
* [Dataset]()
* Python code
### :white_check_mark: Import Libraries
> pandas, pythainlp, gensim, pendas.gensim

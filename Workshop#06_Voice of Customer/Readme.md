# Workshop#06_Voice of Customer

## 	:point_right: Voice of Customer คืออะไร?
**Voice of Customer (VOC) หรือ เสียงจากลูกค้า** คือ การรับฟังข้อมูล ความเห็น การติชม ข้อเสนอแนะของลูกค้าที่มีต่อผลิตภัณฑ์/บริการของบริษัท รวมถึงกระบวนการต่าง ๆ ที่เกี่ยวข้องกับการให้บริการลูกค้า และธุรกิจ หรือความเห็นที่มีความสำคัญกับการพัฒนาผลิตภัณฑ์/บริการของบริษัท ซึ่งปัจจุบัน โดยรวมแบ่งออกเป็น 3 อย่าง ได้แก่
* Contact VOC คือ เสียงหรือความเห็นของลูกค้าที่ได้รับจากการสนทนาโต้ตอบระหว่างบริษัทกับลูกค้า ไม่ว่าจะเป็นช่องทางโทรศัพท์ อีเมล หรือแชทต่าง ๆ
* Social VOC คือ เสียงหรือความเห็นของลูกค้าที่ปรากฎอยู่ใน Social Media Platform ต่าง ๆ เช่น Facebook, Twitter รวมถึงหากบริษัทมีการขายผลิตภัณฑ์/บริการผ่านช่องทางออนไลน์ (e-commerce) ต่าง ๆ
* Survey VOC คือ เสียงหรือความเห็นของลูกค้าที่บริษัทได้จากแบบสอบถาม (Survey) ที่จัดทำขึ้นเพื่อเก็บข้อมูลความเห็นของลูกค้าโดยตรง
  
![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/318b1cc7-69a9-43b7-b9c4-1da99798cbb2)

_ภาพจาก: https://www.talkwalker.com/blog/voice-of-the-customer_



## 	:point_right: Voice of Customer analytics tools
* Sentiment Analysis
* Fake review identification 
* Topic identification
* Net promoter score
* Named entity recognition


## 	:point_right: Topic Modelling with LDA
* **Topic modelling** คือ รูปแบบจำลองทางสถิติที่ใช้สำหรับการหาบทสรุปหรือหัวข้อของบทความหรือข้อมูลที่มีจำนวนมาก ซึ่งในที่นี้จะกล่าวถึง topic modelling รูปแบบหนึ่งที่มีชื่อว่า Latent Dirichlet Allocation (LDA)
* **Latent Dirichlet Allocation** คือ Latent Probabilistic Model ที่ใช้ในการทำ Topic Modelling โดย Topic ที่กล่าวถึงนี้อาจจะเป็น หัวข้อ หรือ theme อะไรบางอย่างที่ซ่อน (hidden) อยู่ในเอกสารและเป็นสิ่งที่เราต้องการ extract มาจากเอกสาร แต่เราและคอมพิวเตอร์ไม่เห็นมันตรง ๆ หมายความ คือ เราจะสามารถบอกได้แค่ว่า คำเหล่านี้อยู่ใน latent topic ที่ 1 2 หรือ 3 และเอกสารนี้มีสัดส่วนของแต่ละ latent topic เท่าไร ซึ่งจะต้องมากำหนดว่าแต่ละ latent topic นั้นกล่าวถึงอะไร เช่น latent topic ที่ 1 กล่าวถึงเรื่องการเมือง latent topic ที่ 2 กล่าวถึงเรื่องกีฬา รวมถึงสามารถระบุได้ว่า topic 1 และ มีคำต่าง ๆ กระจายด้วยความน่าจะเป็นที่เท่าไหร่ ซึ่งเราใช้ความน่าจะเป็นในการอธิบาย topic ต่าง ๆ ที่กระจายอยู่ 

นอกจากนี้ LDA ยังถือเป็น Generative Model หมายความว่า หลังจากที่เราเรียนรู้ความน่าจะเป็นต่าง ๆ จากเอกสาร และคำ ที่เรามีแล้ว การจะสร้างเอกสารใหม่ก็ไม่ใช่เรื่องยากอะไร ก็แค่สุ่ม topic และสุ่มคำจาก topic นั้นมาสร้างเอกสารใหม่ สำหรับวิธีการ generate เอกสารใหม่นั้น เค้าก็เรียกกันกลาง ๆ กันว่า Generative Process

![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/48f2f7bf-7fcf-439e-a26b-c7ceb342622f)

 _แหล่งที่มาข้อมูล: https://pongsakorn-jrc.medium.com/%E0%B8%A3%E0%B8%B9%E0%B9%89%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%81%E0%B8%B1%E0%B8%9A-latent-dirichlet-allocation-part-1-2495acfcda86_

## 	:point_right: Workshop - Topic Modelling with LDA
### :white_check_mark: Dataset
> * [Dataset](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2306_Voice%20of%20Customer/Customer%20Review.csv)
> * [Python code](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/blob/main/Workshop%2306_Voice%20of%20Customer/MADT8101_Voice_of_Customer.ipynb)
> 
> The data set contains information about 30 reviews and ratings by customers for korean restaurant.

### :white_check_mark: Import Libraries
> pandas, pythainlp, gensim, pendas.gensim, pyLDAvis, corpora and models

### :white_check_mark: Exploratory Data Analysis (EDA)
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/8cb1b1e9-8505-4e4c-8215-e5f6ad12a00e)

### :white_check_mark: Tokenize Words and Create Dictionary
> Remove repeated with no meaning word, then split word by comma (,) before put in dictionary (word bucket).
> ```
> removed_words = [' ', ',', ' ,', '\n', 'ร้าน', '(', ')', 'เกาหลี','โค','เรียน','ทาวน์', 'Restaurant', 'Korean','korean','town','Doorae','doorae','DooRae', 'อาหาร','กิน','ทาน']
> ```
>
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/7b359856-cfce-4c08-8a4c-d71ff93a2948)
>
> ```
> print(dictionary.token2id.keys())
> Result:
> dict_keys(['ขายดี', 'ข้าว', 'ชอบ', 'ชั้น', 'ซอล', 'ซอส', 'ตอน', 'ถาม', 'ที่นั่ง', 'บิ', 'ปิ้ง', 'พนักงาน', 'มับ', 'ย่าง', 'รองรับ', 'รู้จัก', 'สดชื่น', 'สามชั้น', 'สไตล์', 'หมัก', 'หมี่', 'หมู', 'หลากหลาย', 'อร่อย', 'ฮิต', 'เกรียม', 'เครื่องเคียง', 'เด้ง', 'เย็น', 'เลือก', 'เส้น', 'Town', 'ความชอบ', 'ชื่อ', 'ดังที่', 'ตั้งอยู่', 'ต้นตำรับ', 'บริเวณ', 'ยังมี', 'ลิ้มลอง', 'สั่ง', 'สุขุมวิท', 'เข้ามา', 'เป็นหนึ่ง', 'เมนู', 'แล้วก็', 'กรอบ', 'ขวัญใจ', 'คิมบับ', 'คุณภาพ', 'จิ', 'ฉับ', 'ซี่โครง', 'ซุป', 'ถูกใจ', 'ธรรมชาติ', 'นัล', 'นัว', 'บริการ', 'บอ', 'ผักสด', 'พิซซ่า', 'ฟี', 'ยอง', 'รส', 'ราคา', 'ร้านอาหาร', 'ล', 'ลาย', 'สะอาด', 'หอมกรุ่น', 'หอย', 'อริ', 'ออกปาก', 'อั้น', 'อู', 'อ่อน', 'ฮอต', 'เซ', 'เด็ด', 'เตา', 'เติม', 'เต้าหู้', 'เบิ้ล', 'เลป', 'แจ่ม', 'แช', 'แน่น', 'แบ', 'แฟน', 'แสง', 'ไฮไลท์', '-', '..', 'กก', 'กลมกล่อม', 'กะ', 'กี่', 'งึบงั้บ', 'จอน', 'จุก', 'ซี', 'ดึ้ง', 'ตลับ', 'ท', 'ทำ', 'ที่มา', 'นุ่ม', 'ป', 'ปัง', 'ฟู้ด', 'รายการ', 'รี', 'ร้อน', 'ว่าที่', 'หอม', 'เกลือ', 'เข้าเนื้อ', 'เนื้อ', 'เปรม', 'เริ่ด', 'เหนียว', 'แป้ง', 'ไข่', 'ไข่ตุ๋น', 'ไส่', 'ไส้', 'ไหนจะ', 'ไหว', 'Asoke', 'กินกัน', 'จาน', 'ซีรีส์', 'ที่อยู่', 'ธร', 'น่ากิน', 'รัน', 'ว', 'อิ้งค์', 'เรื่อง', 'เวิ้ง', 'โปรด', 'ไม่งั้น', 'ๆคน', '.', 'B', 'Doo', 'Q', 'Rae', 'ก', 'กร้าน', 'คน', 'ดี', 'ดีมาก', 'นะคะ', 'ประทับใจ', 'ฟิน', 'ยุ', 'รสชาติ', 'ลอง', 'อี', 'เกะ', 'เค้า', 'เจน', 'เนื้อวัว', 'เนื้อสัตว์', 'เนื้อหมู', 'แนะนำ', 'แปลกใจ', 'ครึ่ง', 'จืดชืด', 'ซ้ำ', 'ดู', 'ติด', 'บ่ายโมง', 'ปกติ', 'ปลาหมึก', 'ผัด', 'ม', 'มบับ', 'สี', 'หวาน', 'เข้มข้น', 'เข้าไป', 'เนื้อสันนอก', 'เอ็น', 'แก้ว', 'แน่นอน', 'โดยรวม', 'ๆๆ', '(คอ', '/', '3', '360', 'กิมจิ', 'คูหา', 'ง.', 'ดีงาม', 'ตัด', 'ถึงกับ', 'บรรยากาศ', 'บุฟเฟ่ต์', 'พน', 'ฟัน', 'ยืน', 'สู้', 'หมูสามชั้น', '฿++)', 'เคี้ยว', 'เพื่อน', 'แคบ', 'แนะ', 'จี', 'ชิ้น', 'ดิบ', 'ดีกว่า', 'ตรงกลาง', 'ทำได้', 'น้ำจิ้ม', 'มีส่วน', 'มุ', 'ส่วนตัว', 'หนา', 'หนึบ', 'หลายอย่าง', 'แนว', 'แฮ', 'ไอ', 'กลาง', 'คาสต์', 'คิว', 'ชั้นสอง', 'บุ', 'ฟิวส์', 'ฟเฟ่', 'ย่าน', 'รา', 'สอบถาม', 'เหมือน', 'แบ่ง', 'โต๊ะ', 'ใกล้เคียง', 'BTS', 'ขับรถ', 'สถานี', 'สะดวก', 'สำหรับ', 'อิ่ม', 'อโศก', 'เดิน', 'เดินทาง', 'ไม่อย่างนั้น', '+', '1', '177', '2', '4', 'A', 'Carte', 'La', 'กระป๋อง', 'กระเทียม', 'ขวด', 'ขา', 'ข้าวสวย', 'คนอื่น', 'ครัว', 'คอ', 'คู่', 'ค่าเสียหาย', 'ด้านนอก', 'ติ', 'ถ้วย', 'ที่นี่', 'นึง', 'น้ำเปล่า', 'บนสุด', 'บันได', 'บันไดเลื่อน', 'บุฟเฟต์', 'พออิ่ม', 'รอบ', 'ริน', 'รู้', 'สีส้ม', 'สูตร', 'ออ', 'ออกมา', 'เซ็ต', 'เทียบ', 'เปลี่ยนไป', 'เริด', 'เสิร์ฟ', 'เหนื่อย', 'เหมือนเดิม', 'เห็ด', 'แข็ง', 'แถม', 'แผ่น', 'แพง', 'แรง', 'แห้ง', 'โค้ก', 'ไหม้', '😂', 'Elite', 'Korea', 'korea', 'กลม', 'การันตี', 'ขนาด', 'ข้อเสีย', 'ข้างนอก', 'คับแคบ', 'จอง', 'ชิม', 'ดอง', 'น้อง', 'มิน่า', 'วงใน', 'สถานที่', 'ส่วนลด', 'เต็ม', 'เลี่ยน', 'เลี้ยง', 'เออ', 'แก้', 'แล้วจึง', 'ใส่', 'ไชเท้า', '60', 'BiBimNengMyeon', 'กลิ่นหอม', 'กำลังดี', 'ก๋วยเตี๋ยว', 'ขอบคุณ', 'ขาว', 'ขึ้นอยู่กับ', 'คลุกเคล้า', 'ความรู้สึก', 'ความสดชื่น', 'คิดเงิน', 'งา', 'จ่าย', 'ชู', 'ตัดเส้น', 'ท้าว', 'น้า', 'น้ำชา', 'น้ำส้มสายชู', 'บาท', 'ปรุงรส', 'ป้าย', 'พนักงานต้อนรับ', 'พนักงานเสิร์ฟ', 'พยางค์', 'พริกป่น', 'มัสตาร์ด', 'มารยาท', 'ยิ้มแย้ม', 'รสเปรี้ยว', 'รีบ', 'รี่ย์', 'ลด', 'ลวก', 'ลูกค้า', 'สอง', 'สีแดง', 'สุก', 'หน้า', 'หัว', 'หาไม่', 'อัธยาศัย', 'เจือจาง', 'เปรี้ยว', 'เผ็ด', 'เรีย', 'เส้นหมี่', 'เหนียวหนึบ', 'แตงกวา', 'แต่ละคน', 'โซน', 'โดด', 'ไข่ต้ม', 'ไช', 'ได้กลิ่น', 'คนเดียว', 'ความสด', 'คุณพ่อ', 'คุณแม่', 'ทานอาหาร', 'ทุ', 'นึกถึง', 'ประจำ', 'ผัก', 'หลายครั้ง', 'เร', 'เวลา', 'แม่', '“', '”', '320', '370', 'กรอ', 'ข้าวโพด', 'จา', 'ด้านใน', 'ตกใจ', 'ตะแกรง', 'ทั่วไป', 'บดี', 'มันฝรั่ง', 'รู้สึก', 'สด', 'สลัด', '฿', 'เด็ก', 'เนย', 'แอปเปิล', 'ไว', '...', '12', ':', 'bts', 'กลางๆ', 'ขวามือ', 'ครั้งแรก', 'คีบ', 'ง', 'ชั้นล่าง', 'ซอย', 'ตัว', 'ต้น', 'ทางออก', 'ปลา', 'มดี', 'สะพานลอย', 'หอ', 'อะ', 'เข้ม', 'เจอ', 'แกง', 'โครง', 'โดดเด่น', 'ใบ', 'Japchae', 'Recommended', 'Senggalbi', 'Yukhoe', 'menu', 'rae', 'restaurant', 'กลิ่น', 'ค่ำ', 'จิ้ม', 'ชื่อเรื่อง', 'ด', 'ปาก', 'พนักงานบริการ', 'ฟรี', 'มมาก', 'ย)', 'ยำ', 'ล้น', 'วุ้นเส้น', 'หนับ', 'หั่น', 'อยากได้', 'อันนี้', 'เครื่อง', 'เช', 'เนื้อสด', 'เรียบร้อย', 'เล่า', 'แครอท', 'แอบ', 'Samgyupsal', 'Sundubu-jjigae', 'bibimbap', 'ค่า', 'ดง', 'ดอก', 'ดิ่ง', 'ทำคะแนน', 'น้อยลง', 'ปุ๊', 'ฟเฟ่ท์', 'ม้วน', 'รอ', 'ลุก', 'วันที่', 'วันอาทิตย์', 'วัว', 'เต็มโต๊ะ', 'เห็บ', 'เอาใจ', 'แกล้ม', 'แนะนำตัว', 'แร', '1250', '25', 'a', 'carte', 'la', 'กระหน่ำ', 'คนละ', 'ถ่าย', 'ทอง', 'ปี', 'ผุด', 'ฟเฟ่ห์', 'ยา', 'สาขา', 'อายุ', 'าน', 'เฉลี่ย', 'เมือง', 'แอปเปิ้ล', 'โควิด', 'คุณภาพดี', 'เคียง', 'แชก็ดี', 'ความจริง', 'นึกออก', 'ห่อ', 'เสริม', 'เหมือนกัน', '', "'", 'BKK', 'It', 'best', 'in', 'my', 'now', 'right', 's', 'คิดถึง', 'งาน', 'ดีจริง', 'ดูแล', 'น', 'พนัก', 'มาหา', 'ละลาย', 'หร่อย', 'ๆ', 'delivery', 'กลับบ้าน', 'ขี้เกียจ', 'ความสะดวก', 'จำนวน', 'ชุด', 'ตามแบบฉบับ', 'น้ำแข็ง', 'บ้าน', 'พวกเรา', 'พอกับ', 'รถ', 'รับได้', 'วันเกิด', 'อยู่แล้ว', 'อาหารจานเดียว', 'เนื้อที่', 'แตกต่าง', 'แพ็ค', 'โทร', 'โอเค', '**', 'คอย', 'ติดตาม', 'ผม', 'สรุป', 'เขียง', 'เสื้อผ้า', 'สัน', 'รับประกัน', 'สม', 'อันดับ', 'เป็นที่นิยม', 'แท้ๆ', 'ให้ได้', '30', 'จอด', 'ที่จอดรถ', 'นั่ง', 'รอง', 'ลาน', 'อีกที', 'อึดอัด', 'เบียด', 'แสตมป์', 'ใต้ดิน', '15', '200', '680', 'กรัม', 'คัด', 'ซัม', 'ตั้งใจ', 'ธรรมดา', 'นาที', 'นิยม', 'บวก', 'พลาซ่า', 'มื้อ', 'รูป', 'ลิสต์', 'วัตถุดิบ', 'สวย', 'สัมผัส', 'สาม', 'ห้า', 'เกินไป', 'เข้ากัน', 'เค็ม', 'แนม', 'แป', 'โครงการ', 'โปรโมท', 'ให้ทาน', 'ความปลอดภัย', 'ความพึงพอใจ', 'ความรับผิดชอบ', 'ความสะอาด', 'ดาว', 'ดาวน์', 'ถ่ายรูป', 'ท่าน', 'บ.', 'ปรับปรุง', 'ปล.', 'ปีก่อน', 'พอใช้', 'รับประทาน', 'สบายใจ', 'สิ่งแปลกปลอม', 'แจ้ง'])
> ```

### :white_check_mark: Topic Modelling
> number of topic = 7
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/64111896-0472-4c96-95a9-201d513df254)

### :white_check_mark: Word Cloud
> Run word cloud and plot graph with thai font, so we have to import libralies (maplotlib.pyplot, numpy) and define paht for thai font as follows:
> ```
> import matplotlib.pyplot as plt
> import numpy as np
> from PIL import Image
> from wordcloud import WordCloud
> ```
> ```
> path = '/content/THSarabunNew.ttf'
> ```
> ```
> wordcloud = WordCloud(font_path='THSarabunNew.ttf',background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=2400, height=1000)
> wordcloud.generate(long_string)
>
> #Display the word cloud using Matplotlib
> plt.figure(figsize=(20, 15))
> plt.imshow(wordcloud, interpolation='bilinear')
> plt.axis('off')  # Hide axis
> plt.show()
> ```
> 
> ![image](https://github.com/Learntogether/MADT8101_Seminar-in-Advanced-Analytics/assets/136689632/647810bb-6e5d-4163-8f83-13dcc9403571)

> 
### :white_check_mark: Pros and Cons of word clouds as visualizations
> _**Pros:**_
> * Reveals the essentials information that what our target (customers) really think about our brand ex. Brand names pop and key words float to the surface.
> * Fast and informative
> * Engaging because word cloud is a visual representation of data tends to have an impact and generate interest amongst the audience.
> 
> _**Cons:**_
> * Size isn't everything, the word cloud is designed to make words stand out according to their size based on their frequency of occurrence, but infact we should consider other factors that might affect to our analysis objectives. 
> * Counting is not comparing 

## 	:triangular_flag_on_post:  Key Benefits and Further Analysis
* Sentiment Analysis: เพื่อแปรผลอารมณ์ และความรู้สึกของลูกค้าที่มีต่อบริษัท ผลิตภัณฑ์/บริการ ตลอดจนกระบวนการต่าง ๆ ที่เกี่ยวข้อง
* Segmentation: แบ่งกลุ่มผู้ใช้งานเพื่อกำหนดวิธีการ/กระบวนการ/หรือผลิตภัณฑ์ เพื่อเข้าถึงกลุ่มลูกค้า (Target audience) แต่ละกลุ่ม รวมถึงการเข้าไป take actions เยียวยากรณีที่ลูกค้าเห็นความรู้สึกทางลบต่อบริษัท

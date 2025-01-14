# Streamlit Application for Customer Behavioral Data
## Objective
* สร้าง streamlit application เพื่อวิเคราะห์ข้อมูลลูกค้าใน supermaket
* Add AI เพื่อให้สามารถถามตอบกับข้อมูลลูกค้่าได้
* ทำ Customer Analysis วิเคราะห์ข้อมูลลูกค้าในแต่ละมุมมอง
* ทำ Clustering เพื่อแนะนำกลุ่มลูกค้าเพื่อจัด Campaign
## Summary Application in one poster
[Profiling Customer Cluster From Behavioral Data 1.pdf](https://github.com/user-attachments/files/18408755/Profiling.Customer.Cluster.From.Behavioral.Data.1.pdf)
## Application Component
มีทั้งหมด 3 หน้า ได้แก่ Overview, Customer Analysis และ Customer Segmentation
1. Overview แสดง sample data ข้อมูลลูกค้า และ statistic table พร้อมทั้ง AI ที่สามารถสอบถามข้อมูลได้ (ดูได้ใน DADS5001_project_page1.pdf)
2. Customer Analysis วิเคราะห์ข้อมูลลูกค้าในแต่ละมุมมอง (ดูได้ใน DADS5001_project_page2.pdf)
3. Customer Segmentation แสดงการแนะนำ campaign ตามกลุ่มลูกค้า (ดูได้ใน DADS5001_project_page3.pdf)
## Dataset (marketing_campaign.csv)
link to original dataset: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data
Dataset ประกอบด้วยข้อมูลดังนี้
* ID: Customer's unique identifier
* Year_Birth: Customer's birth year
* Education: Customer's education level
* Marital_Status: Customer's marital status
* Income: Customer's yearly household income
* Kidhome: Number of children in customer's household
* Teenhome: Number of teenagers in customer's household
* Dt_Customer: Date of customer's enrollment with the company
* Recency: Number of days since customer's last purchase
* Complain: 1 if the customer complained in the last 2 years, 0 otherwise
* Products

* MntWines: Amount spent on wine in last 2 years
* MntFruits: Amount spent on fruits in last 2 years
* MntMeatProducts: Amount spent on meat in last 2 years
* MntFishProducts: Amount spent on fish in last 2 years
* MntSweetProducts: Amount spent on sweets in last 2 years
* MntGoldProds: Amount spent on gold in last 2 years
* Promotion

* NumDealsPurchases: Number of purchases made with a discount
* AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
* AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
* AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
* AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
* AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
* Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
* Place

* NumWebPurchases: Number of purchases made through the company’s website
* NumCatalogPurchases: Number of purchases made using a catalogue
* NumStorePurchases: Number of purchases made directly in stores
* NumWebVisitsMonth: Number of visits to company’s website in the last month

## Streamlit Code
DADS5001_project.py

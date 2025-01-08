import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os
import time
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

new_chat_id = f'{time.time()}'

MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

df = pd.read_csv('https://raw.githubusercontent.com/mayonaise01/DADS5001_Data_Product/refs/heads/main/marketing_campaign.csv',sep="\t")
df = df.dropna()
df['Age'] = 2024 - df['Year_Birth']

# delete row ที่มี income มากกว่า 140000
df = df[df["Income"] <= 140000]
# delete row ที่มี Age มากกว่า 120
df = df[df["Age"] <= 120]
# Group by Marutal Status
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'Single')
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'Couple')
# Group by Education
df['Education'] = df['Education'].replace(['PhD','Master'], 'Postgraduate')
df['Education'] = df['Education'].replace(['Basic'], 'Undergraduate')
df['Education'] = df['Education'].replace(['Graduation','2n Cycle'], 'Graduate')
#Sum of TotalExpenses
df['TotalExpenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
# Calculate the total number of purchases for each customer
df['TotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']

# กำหนดค่าเริ่มต้นใน Session State
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

# กำหนด CSS เพื่อบังคับให้ปุ่มไม่ขึ้นบรรทัดใหม่
st.markdown(
    """
    <style>
    div.stButton > button {
        white-space: nowrap;  /* บังคับข้อความไม่ขึ้นบรรทัดใหม่ */
        overflow: hidden;     /* ตัดข้อความที่ยาวเกิน */
        text-overflow: ellipsis; /* เพิ่ม ... หากข้อความยาวเกิน */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ใช้คอนเทนเนอร์สำหรับ Navigation Bar
with st.container():
    # ปุ่ม Navigation Bar ที่ติดกันในคอลัมน์เดียว
    col1, col2, col3 = st.columns([1, 1, 1])  # ใช้คอลัมน์แบบไม่มีระยะห่างเพิ่มเติม
    with col1:
        home_clicked = st.button("Overview", use_container_width=True)
    with col2:
        about_clicked = st.button("Customer Analysis", use_container_width=True)
    with col3:
        contact_clicked = st.button("Customer Segmentation", use_container_width=True)

    # อัปเดตหน้าใน Session State ตามปุ่มที่กด
    if home_clicked:
        st.session_state["page"] = "Overview"
    elif about_clicked:
        st.session_state["page"] = "Customer Analysis"
    elif contact_clicked:
        st.session_state["page"] = "Customer Segmentation"

# แสดงเนื้อหาตามหน้าที่เลือก
if st.session_state["page"] == "Overview":
    df = df.drop('Z_CostContact', axis=1)
    df = df.drop('Z_Revenue', axis=1)
    st.markdown("""
    <div style="text-align: center; background-color: #ffe599; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: black; font-size: 36px;">Marketing Campaign Analysis🛒</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
    """
    The dataset that contains both demographic information about customers (such as age, education level, etc.) and their 
    purchasing behavior (including money spent across different product categories, frequency of purchases, and recency of activity). 
    The goal is to process this data and use it to create targeted marketing campaigns through customer segmentation. 
    By applying clustering techniques, we can group customers into distinct segments based on similar behaviors and characteristics.\n
    These insights are invaluable for marketing purposes. By understanding the behavior and preferences of each customer segment, 
    you can design tailored campaigns to increase customer retention and boost sales. For example, you could create loyalty programs 
    for high-value customers or offer targeted discounts to encourage repeat purchases from less frequent buyers. Ultimately, the clustering 
    analysis not only helps in improving customer retention but also optimizes the allocation of resources for marketing campaigns, 
    ensuring that the right message reaches the right audience at the right time, thereby maximizing the overall sales potential.
    """
)


    st.image('https://raw.githubusercontent.com/mayonaise01/DADS5001_Data_Product/refs/heads/main/Supermarket profits boomed thanks to the pandemic and inflation.png')
    st.subheader('Sample Data')

    # Display the DataFrame
    config = {
        "Income": st.column_config.NumberColumn("Income ($)"),
    }
    st.dataframe(df.sample(n=50).reset_index(drop=True), column_config=config)

    # Example: Display a summary statistics
    if st.checkbox('Show Summary Statistics'):
        st.write(df.describe())



    ####################################################################################################
    # Set up Google Gemini-Pro AI model
    GOOGLE_API_KEY='AIzaSyDE7kwLshY_Obu-aKWztOv5mxnwUN6qTK8'
    genai.configure(api_key=GOOGLE_API_KEY)

    st.session_state.model = genai.GenerativeModel('gemini-pro')

    def map_role(role):
        if role == "model":
            return "assistant"
        else:
            return role

    def fetch_gemini_response(user_query):
        # Use the session's model to generate a response
        response = st.session_state.chat_session.model.generate_content(
            f"Learn the following 50 sample of customers data in a supermarket:\n{df.sample(n=50).to_markdown()}\n\nHere is the summary statistic of all data:\n{df.describe().to_markdown()}\n\nAnswer the following question: {user_query}"
        )
        print(f"Gemini's Response: {response}")
        return response.parts[0].text


    if "chat_session" not in st.session_state:
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])
        
    # Display the chatbot's title on the page
    st.title("🤖 Ask Gemini-Pro about data")
    st.write("Limitation only 50 samples of data given to generative AI.")
    # Display the chat history
    for msg in st.session_state.chat_session.history:
        with st.chat_message(map_role(msg["role"])):
            st.markdown(msg["content"])

    # Input field for user's message
    user_input = st.chat_input("Ask Gemini-Pro...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        response = fetch_gemini_response(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        # Add user and assistant messages to the chat history
        st.session_state.chat_session.history.append({"role": "user", "content": user_input})
        st.session_state.chat_session.history.append({"role": "model", "content": response})
elif st.session_state["page"] == "Customer Analysis":
    # ส่วนหัวของหน้าจอ
    st.markdown("""
        <div style="text-align: center; background-color: #ffe599; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: black; font-size: 36px;">Customer Analysis Dashboard📊</h1>
            <p style="color: black; font-size: 16px;">Analyze customer data with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)

    # คำนวณ quantile
    quantile_33 = df['TotalExpenses'].quantile(0.33)
    quantile_66 = df['TotalExpenses'].quantile(0.66)

    # สร้างคอลัมน์ใหม่เพื่อแบ่งช่วง TotalExpenses
    def categorize_expenses(expense):
        if expense <= quantile_33:
            return 'Low'
        elif expense <= quantile_66:
            return 'Medium'
        else:
            return 'High'

    df['ExpenseCategory'] = df['TotalExpenses'].apply(categorize_expenses)

    # Layout Section 1: Income and Expense
    st.markdown("### Income and Expense")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.scatter(df, x='Income', y='TotalExpenses', title='Income vs Expense Analysis')
        fig.update_layout(xaxis_title="Income($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        average_income = df['Income'].mean()
        fig = px.histogram(df, x="Income", labels={'count': 'Frequency'}, title='Income Distribution')
        fig.add_vline(x=average_income, line_dash="dash", line_color="red", annotation_text=f"<b>Average: {average_income:,.2f}</b>", annotation_position="top left",annotation_font_size=12)
        fig.update_layout(xaxis_title="Income($)")
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        average_expenses = df['TotalExpenses'].mean()
        fig = px.histogram(df, x="TotalExpenses", labels={'count': 'Frequency'}, title='Expense Distribution')
        fig.add_vline(x=average_expenses, line_dash="dash", line_color="red", annotation_text=f"<b>Average: {average_expenses:,.2f}</b>", annotation_position="top right",annotation_font_size=12)
        # ตั้งค่าให้แกน x เริ่มจาก 0
        fig.update_layout(xaxis=dict(range=[0, df['TotalExpenses'].max() + 1000]))
        fig.update_layout(xaxis_title="TotalExpenses($)")
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Layout Section 2: Age
    st.markdown("### Age")

    average_age = df['Age'].mean()
    fig = px.histogram(df, x="Age", labels={'count': 'Frequency'}, title='Age Distribution')
    fig.add_vline(x=average_age, line_dash="dash", line_color="red", annotation_text=f"<b>Average: {average_age:,.2f}</b>", annotation_position="top right",annotation_font_size=16)
    fig.update_layout(xaxis_title="Age(Years)")
    fig.update_layout(yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Note: Expense Category**
    - **Low**: Expenses($) between 5 - 112 (quantile 0 - 33%)
    - **Medium**: Expenses($) between 112 - 816 (quantile 33 - 66%)
    - **High**: Expenses($) between 816 - 2525 (quantile 66 - 100%)
    """)

    # Layout Section 3: Marital Status
    st.markdown("### Marital Status")
    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(df, x="Marital_Status", color="Marital_Status", labels={'count': 'Frequency'}, title='Marital Status Distribution')
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
        df, 
        x="Marital_Status", 
        color="ExpenseCategory",
        category_orders={"ExpenseCategory": ["Low", "Medium", "High"]},
        labels={'count': 'Frequency'}, 
        title='Marital Status Distribution by Expense Category',
        barmode='stack'  # ตั้งค่าให้ bar ซ้อนกัน
    )
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Layout Section 4: Education Level
    st.markdown("### Education Level")
    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(df, x="Education", color="Education", labels={'count': 'Frequency'}, title='Education Distribution')
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
        df, 
        x="Education", 
        color="ExpenseCategory",
        category_orders={"ExpenseCategory": ["Low", "Medium", "High"]},
        labels={'count': 'Frequency'}, 
        title='Education Distribution by Expense Category',
        barmode='stack'  # ตั้งค่าให้ bar ซ้อนกัน
    )
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Layout Section 5: Kids and Teens
    st.markdown("### KidsAndTeens")
    col1, col2 = st.columns(2)


    with col1:
        df['KidsAndTeens'] = df['Kidhome'] + df['Teenhome']
        KidsAndTeens_counts = df['KidsAndTeens'].value_counts().sort_index()

        fig = px.histogram(df, x="KidsAndTeens", color="KidsAndTeens", category_orders={"KidsAndTeens": [0, 1, 2, 3]}, labels={'count': 'Frequency'}, title='Kids and Teens Distribution')
        fig.update_layout(xaxis_title="No.KidsAndTeens")
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
        df, 
        x="KidsAndTeens", 
        color="ExpenseCategory",
        category_orders={"ExpenseCategory": ["Low", "Medium", "High"]},
        labels={'count': 'Frequency'}, 
        title='KidsAndTeens Distribution by Expense Category',
        barmode='stack'  # ตั้งค่าให้ bar ซ้อนกัน
    )
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 10px; margin-top: 20px; background-color: #ffe599; color: black; border-radius: 10px;">
            <p style="margin: 0;">© 2025 Customer Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
elif st.session_state["page"] == "Customer Segmentation":
    #st.set_page_config(page_title='Clustering', layout='centered')
    #st.title('Customer Segmentation 🛒')
    st.markdown("""
        <div style="text-align: center; background-color: #ffe599; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: black; font-size: 36px;">Customer Segmentation 👨‍👩‍👦‍👦</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
       After evaluating various models( kmeans, ap, meanshift, dbscan, kmodes, etc.), the best silhouette score is shown in kmeans. With high dimension data as ours, we use PCA to reduced dimension of data before clustering process using kmeans.
        """
    )
    st.cache_resource.clear()


    df_encoded = pd.get_dummies(df[['Education','Marital_Status']], dtype=int)
    df = df.drop(['ID','Year_Birth','Dt_Customer','Education','Marital_Status','AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
                'Z_CostContact','Z_Revenue','TotalExpenses','TotalPurchases','Complain', 'Response'],axis=1)

    data = pd.concat([df,df_encoded],axis=1)


    st.subheader('K-Mean with PCA Clustering')

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply PCA to reduce the number of dimensions
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    col1, col2 = st.columns([4, 2], gap='small')
    col1.write('Original Data\n({} dimensions)'.format(len(data.columns)))
    col1.dataframe(data.head(20))

    col2.write('Reduced-dimension\n(2 PCs)')
    data_pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    col2.dataframe(data_pca_df.head(20))

    wcss = []
    for i in range(1,11):
        kmean_pca = KMeans(n_clusters=i, init='k-means++',random_state=42)
        kmean_pca.fit(data_pca)
        wcss.append(np.round(kmean_pca.inertia_,decimals=3))


    st.subheader('WCSS vs Number of Clusters')
    st.write(
    """
    The “elbow” point indicates a strong candidate for the optimal number of clusters at 3.
    """
    )

    fig , ax = plt.subplots()
    ax.plot(range(1,11),wcss,marker = 'o', linestyle =  '--')
    ax.plot([3 for i in range(1,len(wcss)+1)], wcss )
    fig.text(0.4, 0.8, 'Elbow', ha='center', color='darkorange')
    fig.text(0.5, 0.02, 'Number of Clusters', ha='center')
    fig.text(0.01, 0.5, 'WCSS', va='center', rotation='vertical')
    st.pyplot(fig)


    st.subheader('Customer Segments')
    st.write('Merge the 2 PCA components with original data. Apply number of Cluster to generate segments.')

    segm = st.select_slider(
        "WCSS shows elbow at K=3",
        options=[
            2,
            3,
            4,
            5
        ],
        value=3
    )

    if segm==2:
        campaign = {0:'Campaign 1', 1:'Campaign 2'}
    elif segm==3:
        campaign = {0:'Campaign 1', 1:'Campaign 2', 2:'Campaign 3'}
    elif segm==4:
        campaign = {0:'Campaign 1', 1:'Campaign 2', 2:'Campaign 3', 3:'Campaign 4'}
    elif segm==5:
        campaign = {0:'Campaign 1', 1:'Campaign 2', 2:'Campaign 3', 3:'Campaign 4', 4:'Campaign 5'}

    kmean_pca = KMeans(n_clusters=segm, init='k-means++',random_state=42)
    kmean_pca.fit(data_pca)

    df_segm_pca_kmeans = pd.concat([data.reset_index(drop=True),pd.DataFrame(data_pca)],axis=1)
    df_segm_pca_kmeans.columns.values[-2:] = ['Component 1','Component 2'] 
    df_segm_pca_kmeans['Segment K-means PCA'] = kmean_pca.labels_

    df_segm_pca_kmeans['Segments'] = df_segm_pca_kmeans['Segment K-means PCA'].map(campaign)

    config = {
        "Income": st.column_config.NumberColumn("Income ($)"),
    }
    st.dataframe(df_segm_pca_kmeans.head(20), column_config=config)

    fig = px.scatter(
        df_segm_pca_kmeans,
        x="Component 2",
        y="Component 1",
        color="Segments"
    )

    event = st.plotly_chart(fig, key="n3", on_select="rerun")

    st.write(
        """
        Each segmentation contains customer with different behavior. We can drive further to find the significant factors in which campaign can be launched based upon.
        """
    )
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 10px; margin-top: 20px; background-color: #ffe599; color: black; border-radius: 10px;">
            <p style="margin: 0;">© 2025 Customer Segmentation</p>
        </div>
        """, unsafe_allow_html=True)

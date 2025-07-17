import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("C:/Users/ganap/OneDrive/Desktop/Scraped Data/WebScarping/DatamobileAnalysis.csv")

# Clean and prepare data
df = df.dropna(subset=["SmartPhone Price", "Product Name", "Reviews"])

# Convert "SmartPhone Price" and "Reviews" to int after cleaning
df["SmartPhone Price"] = df["SmartPhone Price"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
df["SmartPhone Price"] = pd.to_numeric(df["SmartPhone Price"], errors="coerce")

df["Reviews"] = df["Reviews"].astype(str).str.replace(",", "", regex=False)
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

# Drop rows where conversion failed (if any)
df = df.dropna(subset=["SmartPhone Price", "Reviews"])
df["SmartPhone Price"] = df["SmartPhone Price"].astype(int)
df["Reviews"] = df["Reviews"].astype(int)

# Extract brand from Product Name
df["Brand"] = df["Product Name"].astype(str).apply(lambda x: x.split()[0])

# Most rated mobile
most_rated = df[df["Reviews"] == df["Reviews"].max()]

# Streamlit app
st.title(" Flipkart Smartphone Data Analysis")
st.write("Showing smartphones under ₹50,000 scraped from Flipkart")

# Show DataFrame
st.subheader(" Scraped Smartphone Data")
st.dataframe(df)

# Most rated mobile
st.subheader(" Most Rated Smartphone")
st.write(most_rated[["Product Name", "SmartPhone Price", "Reviews"]])

# Brand-wise count
st.subheader(" Brand-wise Smartphone Count")
brand_count = df["Brand"].value_counts()
st.bar_chart(brand_count)

# Price vs Reviews
st.subheader(" Price vs  Ratings")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="SmartPhone Price", y="Reviews", hue="Brand", palette="Set2", ax=ax)
plt.xlabel("Price (₹)")
plt.ylabel("Number of Reviews")
st.pyplot(fig)

# Optional: Filter by Brand
st.subheader(" Filter by Brand")
selected_brand = st.selectbox("Select Brand", df["Brand"].unique())
filtered_df = df[df["Brand"] == selected_brand]
st.write(f"Smartphones by **{selected_brand}**:")
st.dataframe(filtered_df)

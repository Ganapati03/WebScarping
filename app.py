import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Flipkart Smartphone Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("DatamobileAnalysis.csv")
    
    # Clean and prepare data
    df = df.dropna(subset=["SmartPhone Price", "Product Name", "Reviews"])
    
    # Convert "SmartPhone Price" and "Reviews" to int after cleaning
    df["SmartPhone Price"] = df["SmartPhone Price"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    df["SmartPhone Price"] = pd.to_numeric(df["SmartPhone Price"], errors="coerce")
    
    df["Reviews"] = df["Reviews"].astype(str).str.replace(",", "", regex=False)
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    
    # Drop rows where conversion failed
    df = df.dropna(subset=["SmartPhone Price", "Reviews"])
    df["SmartPhone Price"] = df["SmartPhone Price"].astype(int)
    df["Reviews"] = df["Reviews"].astype(int)

    # --- NEW DATA CLEANING STEP ---
    # Filter out unrealistic prices and review counts
    # Assuming a minimum realistic smartphone price is 1000 and minimum reviews is 1
    df = df[df['SmartPhone Price'] >= 1000]
    df = df[df['Reviews'] >= 1]
    # --- END NEW DATA CLEANING STEP ---
    
    # Extract brand from Product Name
    df["Brand"] = df["Product Name"].astype(str).apply(lambda x: x.split()[0])
    
    # Create price categories - Corrected to include prices above 50000
    df['Price_Category'] = pd.cut(df['SmartPhone Price'], 
                                  bins=[0, 10000, 20000, 30000, 40000, 50000, float('inf')], 
                                  labels=['Under ₹10K', '₹10K-₹20K', '₹20K-₹30K', '₹30K-₹40K', '₹40K-₹50K', '₹50K+'])
    
    # Create review categories
    df['Review_Category'] = pd.cut(df['Reviews'], 
                                   bins=[0, 100, 500, 1000, 5000, float('inf')], 
                                   labels=['Low (0-100)', 'Medium (100-500)', 'High (500-1K)', 'Very High (1K-5K)', 'Extremely High (5K+)'])
    
    return df

# Load data
df = load_data()

# Sidebar for navigation
st.sidebar.title("📱 Navigation")
page = st.sidebar.selectbox("Choose Analysis", [
    "📊 Overview & Summary",
    "🏢 Brand Analysis", 
    "💰 Price Analysis",
    "⭐ Review Analysis",
    "🔍 Advanced Analytics",
    "📈 Interactive Plots",
    "🎯 Recommendations"
])

# Main title
st.markdown('<h1 class="main-header">📱 Flipkart Smartphone Data Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

if page == "📊 Overview & Summary":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Smartphones", len(df))
    with col2:
        st.metric("Total Brands", df['Brand'].nunique())
    with col3:
        st.metric("Avg Price", f"₹{df['SmartPhone Price'].mean():,.0f}")
    with col4:
        st.metric("Total Reviews", f"{df['Reviews'].sum():,}")
    
    # Dataset statistics
    st.subheader("📈 Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Statistics:**")
        price_stats = df['SmartPhone Price'].describe()
        st.write(f"- Min Price: ₹{price_stats['min']:,.0f}")
        st.write(f"- Max Price: ₹{price_stats['max']:,.0f}")
        st.write(f"- Median Price: ₹{price_stats['50%']:,.0f}")
        st.write(f"- Standard Deviation: ₹{price_stats['std']:,.0f}")
    
    with col2:
        st.write("**Review Statistics:**")
        review_stats = df['Reviews'].describe()
        st.write(f"- Min Reviews: {review_stats['min']:,.0f}")
        st.write(f"- Max Reviews: {review_stats['max']:,.0f}")
        st.write(f"- Median Reviews: {review_stats['50%']:,.0f}")
        st.write(f"- Standard Deviation: {review_stats['std']:,.0f}")
    
    # Most rated smartphone
    st.subheader("🏆 Most Rated Smartphone")
    most_rated = df[df["Reviews"] == df["Reviews"].max()]
    st.dataframe(most_rated[["Product Name", "SmartPhone Price", "Reviews", "Brand"]])
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Data quality info
    st.subheader("🔍 Data Quality")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        st.write(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values")
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)

elif page == "🏢 Brand Analysis":
    st.markdown('<h2 class="sub-header">Brand Analysis</h2>', unsafe_allow_html=True)
    
    # Brand distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Brand Distribution")
        brand_count = df["Brand"].value_counts()
        fig = px.bar(x=brand_count.index, y=brand_count.values, 
                     title="Number of Smartphones by Brand",
                     labels={'x': 'Brand', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Brand Market Share")
        fig = px.pie(values=brand_count.values, names=brand_count.index, 
                     title="Brand Market Share")
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand statistics
    st.subheader("📈 Brand Statistics")
    brand_stats = df.groupby('Brand').agg({
        'SmartPhone Price': ['mean', 'min', 'max', 'count'],
        'Reviews': ['mean', 'sum']
    }).round(2)
    
    brand_stats.columns = ['Avg Price', 'Min Price', 'Max Price', 'Phone Count', 'Avg Reviews', 'Total Reviews']
    brand_stats = brand_stats.sort_values('Phone Count', ascending=False)
    st.dataframe(brand_stats)
    
    # Brand price comparison
    st.subheader("💰 Brand Price Comparison")
    fig = px.box(df, x='Brand', y='SmartPhone Price', 
                 title="Price Distribution by Brand")
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)
    
    # Brand review comparison
    st.subheader("⭐ Brand Review Comparison")
    fig = px.box(df, x='Brand', y='Reviews', 
                 title="Review Distribution by Brand")
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)
    
    # Top brands by average price
    st.subheader("🏆 Top Brands by Average Price")
    top_brands_price = df.groupby('Brand')['SmartPhone Price'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(x=top_brands_price.index, y=top_brands_price.values,
                 title="Top 10 Brands by Average Price",
                 labels={'x': 'Brand', 'y': 'Average Price (₹)'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "💰 Price Analysis":
    st.markdown('<h2 class="sub-header">Price Analysis</h2>', unsafe_allow_html=True)
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Distribution")
        fig = px.histogram(df, x='SmartPhone Price', nbins=30,
                           title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Price Categories")
        price_cat_count = df['Price_Category'].value_counts().sort_index() # Sort to ensure correct order
        fig = px.bar(x=price_cat_count.index, y=price_cat_count.values,
                     title="Smartphones by Price Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Reviews correlation
    st.subheader("📈 Price vs Reviews Correlation")
    fig = px.scatter(df, x='SmartPhone Price', y='Reviews', color='Brand',
                     title="Price vs Reviews Correlation",
                     hover_data=['Product Name'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Price statistics by category
    st.subheader("📊 Price Statistics by Category")
    price_stats_by_cat = df.groupby('Price_Category').agg({
        'SmartPhone Price': ['count', 'mean', 'min', 'max'],
        'Reviews': 'mean'
    }).round(2)
    price_stats_by_cat.columns = ['Count', 'Avg Price', 'Min Price', 'Max Price', 'Avg Reviews']
    st.dataframe(price_stats_by_cat)
    
    # Most expensive and cheapest phones
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💎 Most Expensive Phones")
        expensive = df.nlargest(5, 'SmartPhone Price')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews']]
        st.dataframe(expensive)
    
    with col2:
        st.subheader("💰 Most Affordable Phones")
        affordable = df.nsmallest(5, 'SmartPhone Price')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews']]
        st.dataframe(affordable)

elif page == "⭐ Review Analysis":
    st.markdown('<h2 class="sub-header">Review Analysis</h2>', unsafe_allow_html=True)
    
    # Review distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Review Distribution")
        fig = px.histogram(df, x='Reviews', nbins=30,
                           title="Review Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Review Categories")
        review_cat_count = df['Review_Category'].value_counts().sort_index() # Sort to ensure correct order
        fig = px.bar(x=review_cat_count.index, y=review_cat_count.values,
                     title="Smartphones by Review Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Most reviewed phones
    st.subheader("🏆 Most Reviewed Smartphones")
    most_reviewed = df.nlargest(10, 'Reviews')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews']]
    st.dataframe(most_reviewed)
    
    # Review statistics by brand
    st.subheader("📊 Review Statistics by Brand")
    review_stats_by_brand = df.groupby('Brand').agg({
        'Reviews': ['count', 'mean', 'sum', 'max']
    }).round(2)
    review_stats_by_brand.columns = ['Phone Count', 'Avg Reviews', 'Total Reviews', 'Max Reviews']
    review_stats_by_brand = review_stats_by_brand.sort_values('Total Reviews', ascending=False)
    st.dataframe(review_stats_by_brand)
    
    # Brand with most total reviews
    st.subheader("🏆 Brands with Most Total Reviews")
    total_reviews_by_brand = df.groupby('Brand')['Reviews'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=total_reviews_by_brand.index, y=total_reviews_by_brand.values,
                 title="Top 10 Brands by Total Reviews")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Advanced Analytics":
    st.markdown('<h2 class="sub-header">Advanced Analytics</h2>', unsafe_allow_html=True)
    
    # Correlation matrix
    st.subheader("📊 Correlation Matrix")
    corr_matrix = df[['SmartPhone Price', 'Reviews']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Value for money analysis
    st.subheader("💎 Value for Money Analysis")
    df['Reviews_per_1000_Rupees'] = df['Reviews'] / (df['SmartPhone Price'] / 1000)
    value_phones = df.nlargest(10, 'Reviews_per_1000_Rupees')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews', 'Reviews_per_1000_Rupees']]
    st.dataframe(value_phones)
    
    # Price vs Reviews by Price Category
    st.subheader("📈 Price vs Reviews by Price Category")
    fig = px.scatter(df, x='SmartPhone Price', y='Reviews', color='Price_Category',
                     title="Price vs Reviews by Price Category",
                     hover_data=['Product Name', 'Brand'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand popularity vs price strategy
    st.subheader("🎯 Brand Strategy Analysis")
    brand_strategy = df.groupby('Brand').agg({
        'SmartPhone Price': 'mean',
        'Reviews': 'mean',
        'Product Name': 'count'
    }).round(2)
    brand_strategy.columns = ['Avg Price', 'Avg Reviews', 'Phone Count']
    brand_strategy = brand_strategy[brand_strategy['Phone Count'] >= 3]  # Filter brands with at least 3 phones
    
    fig = px.scatter(brand_strategy, x='Avg Price', y='Avg Reviews', 
                     size='Phone Count', hover_name=brand_strategy.index,
                     title="Brand Strategy: Average Price vs Average Reviews")
    st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution by brand (violin plot)
    st.subheader("🎻 Price Distribution by Brand (Violin Plot)")
    brands_with_multiple = df['Brand'].value_counts()[df['Brand'].value_counts() >= 3].index
    df_filtered = df[df['Brand'].isin(brands_with_multiple)]
    
    fig = px.violin(df_filtered, x='Brand', y='SmartPhone Price',
                    title="Price Distribution by Brand (Brands with 3+ phones)")
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Interactive Plots":
    st.markdown('<h2 class="sub-header">Interactive Visualizations</h2>', unsafe_allow_html=True)
    
    # Interactive filters
    st.sidebar.subheader("🔧 Filters")
    
    # Brand filter
    selected_brands = st.sidebar.multiselect("Select Brands", 
                                             options=df['Brand'].unique(),
                                             default=df['Brand'].unique()[:5])
    
    # Price range filter
    price_range = st.sidebar.slider("Price Range (₹)", 
                                    min_value=int(df['SmartPhone Price'].min()),
                                    max_value=int(df['SmartPhone Price'].max()),
                                    value=(int(df['SmartPhone Price'].min()), 
                                           int(df['SmartPhone Price'].max())))
    
    # Review range filter
    review_range = st.sidebar.slider("Review Range", 
                                     min_value=int(df['Reviews'].min()),
                                     max_value=int(df['Reviews'].max()),
                                     value=(int(df['Reviews'].min()), 
                                            int(df['Reviews'].max())))
    
    # Apply filters
    filtered_df = df[
        (df['Brand'].isin(selected_brands)) &
        (df['SmartPhone Price'] >= price_range[0]) &
        (df['SmartPhone Price'] <= price_range[1]) &
        (df['Reviews'] >= review_range[0]) &
        (df['Reviews'] <= review_range[1])
    ]
    
    st.write(f"Showing {len(filtered_df)} smartphones based on filters")
    
    # Interactive scatter plot
    st.subheader("📊 Interactive Scatter Plot")
    fig = px.scatter(filtered_df, x='SmartPhone Price', y='Reviews', 
                     color='Brand', size='Reviews',
                     hover_data=['Product Name'],
                     title="Interactive Price vs Reviews")
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("🌐 3D Visualization")
    filtered_df['Price_Rank'] = filtered_df['SmartPhone Price'].rank(ascending=False)
    fig = px.scatter_3d(filtered_df, x='SmartPhone Price', y='Reviews', 
                        z='Price_Rank', color='Brand',
                        title="3D: Price vs Reviews vs Price Rank",
                        hover_data=['Product Name'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Filtered data table
    st.subheader("📋 Filtered Data")
    st.dataframe(filtered_df[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews']])

elif page == "🎯 Recommendations":
    st.markdown('<h2 class="sub-header">Smart Recommendations</h2>', unsafe_allow_html=True)
    
    # Best value phones
    st.subheader("💰 Best Value Phones")
    df['Value_Score'] = (df['Reviews'] / df['SmartPhone Price'] * 1000).round(2)
    best_value = df.nlargest(10, 'Value_Score')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews', 'Value_Score']]
    st.dataframe(best_value)
    
    # Top picks by price range
    st.subheader("🏆 Top Picks by Price Range")
    
    # Ensure categories are ordered for consistent display
    price_categories_order = ['Under ₹10K', '₹10K-₹20K', '₹20K-₹30K', '₹30K-₹40K', '₹40K-₹50K', '₹50K+']
    for category in price_categories_order:
        if category in df['Price_Category'].cat.categories: # Check if category exists in the data
            st.write(f"**{category}:**")
            category_df = df[df['Price_Category'] == category]
            top_in_category = category_df.nlargest(3, 'Reviews')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews']]
            if not top_in_category.empty:
                st.dataframe(top_in_category)
            else:
                st.write("No phones in this category.")
    
    # Brand recommendations
    st.subheader("🏢 Brand Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Most Popular Brands (by total reviews):**")
        popular_brands = df.groupby('Brand')['Reviews'].sum().sort_values(ascending=False).head(5)
        for brand, reviews in popular_brands.items():
            st.write(f"- {brand}: {reviews:,} total reviews")
    
    with col2:
        st.write("**Premium Brands (by average price):**")
        premium_brands = df.groupby('Brand')['SmartPhone Price'].mean().sort_values(ascending=False).head(5)
        for brand, price in premium_brands.items():
            st.write(f"- {brand}: ₹{price:,.0f} average price")
    
    # Personalized recommendations
    st.subheader("🎯 Find Your Perfect Phone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input("Your Budget (₹)", min_value=5000, max_value=int(df['SmartPhone Price'].max()), value=20000, step=1000)
        preferred_brands = st.multiselect("Preferred Brands (optional)", df['Brand'].unique())
    
    with col2:
        min_reviews = st.number_input("Minimum Reviews", min_value=0, max_value=int(df['Reviews'].max()), value=100)
        show_top_n = st.slider("Show top N recommendations", 1, 20, 5)
    
    # Generate recommendations
    recommendation_df = df[df['SmartPhone Price'] <= budget]
    
    if preferred_brands:
        recommendation_df = recommendation_df[recommendation_df['Brand'].isin(preferred_brands)]
    
    recommendation_df = recommendation_df[recommendation_df['Reviews'] >= min_reviews]
    
    if not recommendation_df.empty:
        recommendations = recommendation_df.nlargest(show_top_n, 'Reviews')[['Product Name', 'Brand', 'SmartPhone Price', 'Reviews']]
        st.write(f"**Top {show_top_n} Recommendations for your criteria:**")
        st.dataframe(recommendations)
    else:
        st.write("No phones found matching your criteria. Try adjusting your filters.")

# Footer
st.markdown("---")
st.markdown("📱 **Flipkart Smartphone Analysis Dashboard** | Built with Streamlit & Plotly")
st.markdown("*Data scraped from Flipkart for smartphones under ₹50,000*")

import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px
import numpy as np
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="The Shift to EVs Among Malaysian Drivers Dashboard", layout="wide")

# Title
st.title("The Shift to EVs Among Malaysian Drivers Dashboard")

# Define the folder containing the CSV files
folder_path = 'C:/Users/Thinktechniq/Desktop/Forward College Data Analytics/Capstone Project/Datasets (MY Vehicle Registration_2015 to 2025)'

# Get list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Load and combine all CSVs
@st.cache_data
def load_data(file_list):
    df_list = [pd.read_csv(file) for file in file_list]
    combined = pd.concat(df_list, ignore_index=True)
    
    # Normalize column names
    combined.columns = combined.columns.str.lower().str.strip()
    
    # Convert 'date_reg' to datetime and create 'year_reg'
    if 'date_reg' in combined.columns:
        combined['date_reg'] = pd.to_datetime(combined['date_reg'], errors='coerce')
        combined['year_reg'] = combined['date_reg'].dt.year
        # Drop NaN years and convert to int
        combined = combined[combined['year_reg'].notna()].copy()
        combined['year_reg'] = combined['year_reg'].astype(int)
    else:
        st.warning("'date_reg' column is missing. 'year_reg' cannot be created.")

    return combined

# Load data
combined_df = load_data(csv_files)

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Homepage", "Data View", "Data Analysis"])

# Homepage
if page == "Homepage":
    st.header("Welcome to the Malaysian Automobile Market Trends Dashboard")
    st.write("""
    This dashboard provides an interactive view of Malaysia’s vehicle registration data from 2015 to 2025,
    with a focus on electric vehicle (EV) adoption trends.

    Use the sidebar to navigate to:
    - **Data View** to explore raw records, and
    - **Data Analysis** to view market insights.
    """)

# Data View
if page == "Data View":
    st.header("Raw Data View")
    st.write("You can preview the combined dataset below:")
    row_num = st.slider("Number of Rows to View", 10, len(combined_df))
    st.dataframe(combined_df.head(row_num))

# New section: Registrations by Year
    st.subheader("Number of Registrations by Year")
    reg_by_year=combined_df['year_reg'].value_counts().sort_index()
    reg_by_year_df=reg_by_year.reset_index()
    reg_by_year_df.columns=['Year','Number of Registrations']
    st.table(reg_by_year_df)

# Data Analysis
if page == "Data Analysis":
    st.header("Automobile Market Trends Analysis (2015-2025)")

    # Filter the dataframe for only EV cars
    fuel_type='electric'
    df_EV_car_type=combined_df[combined_df["fuel"]==fuel_type]

    # Pivot table showing registrations by car type and year
    pivot_EV_type=pd.pivot_table(
        data=df_EV_car_type,
        index='type',
        columns='year_reg',
        aggfunc='size',
        fill_value=0
    )

    # ✅ Rename car types
    type_rename_map = {
        'jip': 'SUV',
        'motokar': 'Sedan',
        'motokar_pelbagai_utiliti': 'MPV',
        'pick_up': 'Pick-Up',
        'window_van': 'Window Van'
        }
    pivot_EV_type=pivot_EV_type.rename(index=type_rename_map)

    # Pivot table showing the proportion of registrations by car type from 2015 to 2025
    pivot_proportion_EV_type=pivot_EV_type.div(pivot_EV_type.sum(axis=0),axis=1)
    pivot_percentage_EV_type=pivot_proportion_EV_type*100

    # Plot proportion of registrations from 2015 to 2025 (by EV car type)
    top_EV_car_types=pivot_proportion_EV_type.sum(axis=1).sort_values(ascending=False).head(5).index
    top_5_EV_type_df=pivot_proportion_EV_type.loc[top_EV_car_types]
    df_plot_type=top_5_EV_type_df.T.reset_index().melt(
        id_vars='year_reg',
        var_name='type',
        value_name='proportion'
    )

    # Interactive line chart
    fig=px.line(
        df_plot_type,
        x='year_reg',
        y='proportion',
        color='type',
        title='Market Share Trends by EV Car Types (2015-2025)',
        markers=True,
        hover_name='type',
        hover_data={'proportion':':.2%'}
    )

    fig.update_layout(
        xaxis_title='Type',
        yaxis_title='Proportion of Registrations',
        yaxis_tickformat='.0%',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
By EV car type, sedans and SUVs are the two most popular car types. The most recent data in 2025 shows that the trend is shifting towards SUVS - with 60.25% of the vehicles registered being EVs and 26.81% being sedans.
""")

#--
# Pivot table showing registrations by fuel type and year 
    pivot_fuel=pd.pivot_table(
        data=combined_df,
        index="fuel",
        columns="year_reg",
        values="date_reg",
        aggfunc="count",
        fill_value=0
    )

# Pivot table showing the proportion of registrations by fuel type from 2015 to 2025
    pivot_proportion_fuel=pivot_fuel.div(pivot_fuel.sum(axis=0),axis=1)
    pivot_percentage_fuel=pivot_proportion_fuel*100
    
# Plot proportion of registrations from 2015 to 2025 (by fuel types)
    top_fuel = pivot_proportion_fuel.sum(axis=1).sort_values(ascending=False).head(5).index
    pivot_proportion_fuel.loc[top_fuel].T.plot(figsize=(10,6), marker='o')

# Interactive line chart
    df_plot_fuel = (
        pivot_proportion_fuel.loc[top_fuel]
        .T
        .reset_index()
        .melt(id_vars='year_reg', var_name='fuel', value_name='proportion')
    )

    fig = px.line(
        df_plot_fuel,
        x='year_reg',
        y='proportion',
        color='fuel',
        title='Interactive Market Share Trends by Fuel Type (2015–2025)',
        markers=True,
        hover_name='fuel',
        hover_data={'proportion': ':.2%'}
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Proportion of Registrations',
        yaxis_tickformat='.0%',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
When looking at fuel types, electric vehicles (EVs) saw the most dramatic rise — from a negligible 0.01% in 2015 to 4.32% in 2025. Notably, its share began to jump only after 2020. Its share remained 0.01% or lower from 2015 to 2020.
""")

    # Pivot table: registrations by EV maker and year
    # First, filter the dataframe for only EV cars
    fuel_type='electric'
    df_EV_maker=combined_df[combined_df["fuel"]==fuel_type]

    pivot_EV_maker=pd.pivot_table(
        data=df_EV_maker,
        index="maker",
        columns="year_reg",
        aggfunc="size",
        fill_value=0
        )

    # Proportions per year
    pivot_proportion_EV_maker=pivot_EV_maker.div(pivot_EV_maker.sum(axis=0),axis=1)

    # Select top 10 EV makers
    top_EV_makers=pivot_proportion_EV_maker.sum(axis=1).sort_values(ascending=False).head(10).index

    # Prepare data for Plotly
    df_plot_EV_maker=(
        pivot_proportion_EV_maker.loc[top_EV_makers]
        .T
        .reset_index()
        .melt(id_vars='year_reg',var_name='maker',value_name='proportion')
        )

    # Plot
    fig=px.line(
        df_plot_EV_maker,
        x='year_reg',
        y='proportion',
        color='maker',
        title='Market Share Trends for Top 10 EV Makers (2015-2025)',
        markers=True,
        hover_name='maker',
        hover_data={'proportion':':.2%'}
        )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Proportion of Registrations',
        yaxis_tickformat='.0%'
        )

    st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
The EV market experienced considerable shifts in brand dominance. In 2015, Renault held a commanding 81.82% of the EV market (from just 45 cars), but by 2025, the brand had vanished from the EV scene. Other early leaders — such as Nissan and Porsche, which held 72.73% (2019) and 69.65% (2021) market shares respectively — also saw sharp declines. By 2025, Nissan’s share fell to 0.02%, while Porsche dropped to 2.09%.
""")

    # Filter EV dataset
    df_ev = combined_df[combined_df['fuel'] == 'electric']
    df_ev['model'] = df_ev['model'].astype(str).str.strip().str.title()
    
    # Pivot table: registrations by model and year
    pivot_model = pd.pivot_table(
        data=df_ev,
        index="model",
        columns="year_reg",
        aggfunc="size",
        fill_value=0
        )

    # Proportions
    pivot_proportion_model = pivot_model.div(pivot_model.sum(axis=0), axis=1)
    
    # Get top 15 models by total proportion
    top_15_models = pivot_proportion_model.sum(axis=1).sort_values(ascending=False).head(15).index

    # Remove 'Unknown' from list if present
    top_15_models = top_15_models[~top_15_models.str.contains("Unknown", case=False, na=False)]
    
    # Filter pivot table to only top 15 models (display registration counts)
    top_15_models_pivot=pivot_model.loc[top_15_models]

    # Filter pivot table to only top 15 models (display proportion)
    top_15_models_proportion_pivot=pivot_proportion_model.loc[top_15_models]
    print(top_15_models_proportion_pivot)
    
    # Prepare data for interactive plot
    df_plot_model = (
        pivot_proportion_model.loc[top_15_models]
        .T
        .reset_index()
        .melt(id_vars='year_reg', var_name='model', value_name='proportion')
    )

    # Interactive line chart
    fig = px.line(
        df_plot_model,
        x='year_reg',
        y='proportion',
        color='model',
        title='Market Share Trends for Top 15 EV Models',
        markers=True,
        hover_name='model',
        hover_data={'proportion': ':.2%'}
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Proportion of Registrations',
        yaxis_tickformat='.0%',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
In the EV space, BYD Atto 3 led in sales from its launch until December 2024, when Proton e.MAS 7 entered the market. In 2025, the e.MAS 7 became Malaysia’s best-selling EV model, with 4,003 registrations (23.35%), followed by BYD Atto 3 (1,701 units, 9.92%) and Tesla Model Y (1,571 units, 9.16%).
""")
        

#--
    # Pivot table: registrations by Chinese vs. Foreign brands and year
    # First, assign the EV car makers (brands) the "Chinese" and "Foreign" region categories according to their country of origin
    ## Define "Chinese", "Local" categories
    chinese_brands = [
        "BAIC", "BAW", "BYD", "Chery", "Denza", "Dong Feng", "Foday", "Foton",
        "GAC", "GAC Aion", "GAC Hyptec", "Great Wall", "Higer", "JAC", "Jetour",
        "JMC", "King Long Xiamen", "Leapmotor", "Maxus", "Neta", "Shineray",
        "Smart", "Xinneng", "Xpeng", "Zeekr", "ZXAuto"
    ]
    local_brands = ["Proton", "Perodua"]

    ## Filter the data for EVs only
    fuel_type="electric"
    df_EV_maker=combined_df[combined_df["fuel"]==fuel_type].copy()

    ## Assign region category to each maker
    def categorize_maker(maker):
        if maker in chinese_brands:
            return 'China'
        elif maker in local_brands:
            return 'Malaysia'
        else:
            return 'Foreign'

    df_EV_maker['origin']=df_EV_maker['maker'].apply(categorize_maker)

    pivot_EV_origin=pd.pivot_table(
        data=df_EV_maker,
        index='origin',
        columns='year_reg',
        aggfunc='size',
        fill_value=0
    )

    # Proportions per year
    pivot_proportion_EV_origin = pivot_EV_origin.div(pivot_EV_origin.sum(axis=0),axis=1)

    # Prepare data for Plotly
    df_plot_EV_origin=(
        pivot_proportion_EV_origin
        .T
        .reset_index()
        .melt(id_vars='year_reg',var_name='origin',value_name='proportion')  
        )

    # Plot
    fig = px.line(
        df_plot_EV_origin,
        x='year_reg',
        y='proportion',
        color='origin',
        title='EV Market Share by Brand Origin (2015–2025)',
        markers=True,
        hover_name='origin',
        hover_data={'proportion': ':.2%'}
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Proportion of Registrations',
        yaxis_tickformat='.0%'
    )

    st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
Foreign EV brands—such as Tesla, Renault, and Nissan—were the early leaders in Malaysia’s nascent EV market. However, this changed with the entry of Chinese brands like BYD and Chery in late 2022. Since 2023, Chinese EV brands have taken the lead, dominating the Malaysian EV market.
""")

#--

    # Map the top 10 EV models with their prices

    model_price_dict = {
    'Leaf': 168888,
    'Zoe': 165000,
    'Taycan': 575000,
    'Cooper': 193888,
    'Model 3':	189000,
    'Atto 3': 123800,
    'iX': 385430,
    'Ativa': 71200,
    'Model X': 238000,
    'Twizy': 71888,
    'Model S': 339000
    }

    # Create cleaned version of model_price_dict
    cleaned_model_price_dict = {
        k.strip().replace('\u202f', ' ').replace('\xa0', ' '): v
        for k, v in model_price_dict.items()
    }
    
    # Pivot table for EV models by year
    pivot_ev_model = pd.pivot_table(
        data=df_ev,
        index='model',
        columns='year_reg',
        aggfunc='size',
        fill_value=0
    )
    pivot_proportion_ev = pivot_ev_model.div(pivot_ev_model.sum(axis=0), axis=1)

# Get top 10 EV models (based on total market share from 2015 to 2025)
    top_10_ev_models = (
        pivot_proportion_ev.sum(axis=1)
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )
    
    # Count registrations for each top EV model
    top_ev_model_counts = combined_df[combined_df['model'].isin(top_10_ev_models)]['model'].value_counts()

    # Normalize model names (remove weird spaces and non-breaking spaces)
    top_ev_model_counts.index = (
        top_ev_model_counts.index
        .str.strip()
        .str.replace('\u202f', ' ')
        .str.replace('\xa0', ' ')
    )

    # Map model names to their prices
    top_ev_model_prices = top_ev_model_counts.index.to_series().map(cleaned_model_price_dict)

    # Combine into a DataFrame
    ev_model_table = pd.DataFrame({
        'Registrations': top_ev_model_counts.values,
        'Price (RM)': top_ev_model_prices.values
        }, index=top_ev_model_counts.index)

# Optional: sort by registration count
    ev_model_table = ev_model_table.sort_values(by='Registrations', ascending=False)

# Display the table in Streamlit
    st.subheader("Top 10 EV Models: Registrations and Prices")
    st.dataframe(ev_model_table.style.format({"Price (RM)": "RM {:,.0f}"}))

# Compute weighted median price
    top_ev_model_weighted_median_price = np.median(top_ev_model_prices)

# Display weighted median price in Streamlit
    st.markdown(f"Weighted median price among top 10 EV models: RM {top_ev_model_weighted_median_price:,.2f}")

#--

# WordCloud 1: Malaysian drivers' sentiment on EV car purchase

    st.subheader("Public Sentiment on EV Cars (WordCloud)")

    # Load the image
    image = Image.open(r"C:\Users\Thinktechniq\Desktop\Forward College Data Analytics\Capstone Project\EV_Wordcloud 1.png") 

    # Display it in Streamlit
    st.image(image, caption="WordCloud of Malaysian EV-related Reddit Thread: Are EV cars practical in Malaysia?")

    st.markdown("""
    Keywords: Charger, cost, electricity, fuel, petrol
    """)
    
# WordCloud 2: Malaysian drivers' sentiment on EV car purchase

    # Load the image
    image = Image.open(r"C:\Users\Thinktechniq\Desktop\Forward College Data Analytics\Capstone Project\EV_Wordcloud 2.png")  
    
    # Display it in Streamlit
    st.image(image, caption="WordCloud of Malaysian EV-related Reddit Thread: EV owners, what are your thoughts after using them after few years?")

    st.markdown("""
    Keywords: Charger, charging, charge, home, time, cost
    """)

# WordCloud 3: Malaysian drivers' sentiment on EV car purchase

    # Load the image
    image = Image.open(r"C:\Users\Thinktechniq\Desktop\Forward College Data Analytics\Capstone Project\EV_Wordcloud 3.png")  
    
    # Display it in Streamlit
    st.image(image, caption="WordCloud of Malaysian EV-related Reddit Thread: How long more will ICE cars stay relevant in Malaysia?")

    st.markdown("""
    Keywords: Time, longer, stations, years, fast, battery
    """)

# WordCloud 4: Malaysian drivers' sentiment on EV car purchase

    # Load the image
    image = Image.open(r"C:\Users\Thinktechniq\Desktop\Forward College Data Analytics\Capstone Project\EV_Wordcloud 4.png")  
    
    # Display it in Streamlit
    st.image(image, caption="WordCloud of Malaysian EV-related Reddit Thread: EV worth it?")

    st.markdown("""
    Keywords: Save, cost, charger
    """)


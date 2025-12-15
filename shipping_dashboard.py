import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Maritime CO2 Footprint & Efficiency",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚢 Maritime Carbon Footprint & Fleet Efficiency Dashboard")
st.markdown("---")

# --- 1. Data Loading and Merging ---

@st.cache_data
def load_and_merge_data():
    """Loads all necessary CSV files and merges them for dashboard use."""
    try:
        # Load core data (EU MRV cleaned)
        df_mrv = pd.read_csv("./eda6_outputs/eu_mrv_cleaned_sample.csv")
        # Load ranking results
        df_ranking = pd.read_csv("./eda6_outputs/step9_full_ship_efficiency_ranking.csv")
        # Load prediction results
        df_predictions = pd.read_csv("./eda6_outputs/step10_model_predictions.csv")
        
        # Merge Ranking data with MRV data
        df_merged = df_ranking.merge(
            df_mrv[['IMO', 'ShipType', 'DWT', 'DistanceNm', 'CO2_tonnes']], 
            on='IMO', 
            how='left'
        )
        
        # Merge predictions on input features (DistanceNm and DWT) due to missing IMO in pred file
        df_predictions.rename(columns={'DistanceNm': 'DistanceNm_Pred', 'DWT': 'DWT_Pred'}, inplace=True)
        
        df_merged = df_merged.merge(
            df_predictions,
            left_on=['DistanceNm', 'DWT'],
            right_on=['DistanceNm_Pred', 'DWT_Pred'],
            how='left',
            suffixes=('_MRV', '_Pred')
        )
        
        # Final cleanup and renaming
        df_merged.rename(columns={'CO2_per_nm': 'CO2_per_nm_Rank'}, inplace=True)
        df_merged['CO2_per_nm_Rank'] = df_merged['CO2_per_nm_Rank'].replace([np.inf, -np.inf], np.nan)
        df_merged['DWT_Tonnes'] = df_merged['DWT']
        
        df_merged.drop(columns=['DistanceNm_Pred', 'DWT_Pred', 'Fuel_tonnes'], inplace=True, errors='ignore')
        
        return df_merged
        
    except FileNotFoundError as e:
        st.error(f"Error: One or more required CSV files not found: {e}")
        return None

df = load_and_merge_data()

if df is not None:
    
    # --- 2. Sidebar Filter ---
    st.sidebar.header("Filter Fleet Data")
    
    ship_types = sorted(df['ShipType'].dropna().unique())
    selected_ship_types = st.sidebar.multiselect(
        "Select Ship Type(s):",
        options=ship_types,
        default=ship_types 
    )
    
    df_filtered = df[df['ShipType'].isin(selected_ship_types)].copy()

    # ====================================================================
    # 3. KPI Display (Key Performance Indicators)
    # ====================================================================
    kpi1, kpi2, kpi3 = st.columns(3)

    # KPI 1: Total CO2 Emissions
    total_co2 = df_filtered['CO2_tonnes'].sum()
    kpi1.metric(
        label="Total CO₂ Emissions (tonnes)", 
        value=f"{total_co2:,.0f}"
    )

    # KPI 2: Total Vessels
    total_vessels = df_filtered['IMO'].nunique()
    kpi2.metric(
        label="Total Vessels in Fleet", 
        value=f"{total_vessels:,}"
    )

    # KPI 3: Average Fleet Efficiency
    avg_efficiency = df_filtered['CO2_per_nm_Rank'].mean()
    kpi3.metric(
        label=r"Avg. Efficiency ($\mathbf{CO}_2 \text{ per nm}$)", 
        value=f"{avg_efficiency:.2f}"
    )
    st.markdown("---")


    # --- 4. Dashboard Layout (2x2 Grid) ---
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Chart 1: Ship Efficiency Ranking (Top 20)
    with col1:
        st.header(r"1. Efficiency Ranking ($\mathbf{CO}_2 \text{ per nm}$)")
        df_ranking_chart = df_filtered.sort_values(by='CO2_per_nm_Rank', ascending=True).head(20)
        fig1 = px.bar(
            df_ranking_chart, x='CO2_per_nm_Rank', y='ShipName', orientation='h',
            title='Top 20 Most Efficient Ships', color='CO2_per_nm_Rank',
            color_continuous_scale=px.colors.sequential.Plasma_r
        )
        fig1.update_layout(yaxis={'autorange': "reversed"}, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Efficiency Distribution (Box Plot)
    with col2:
        st.header("2. Efficiency Distribution by Type")
        fig2 = px.box(
            df_filtered, x='ShipType', y='CO2_per_nm_Rank', points="outliers",
            title='CO2/nm Distribution Across Ship Types'
        )
        fig2.update_layout(xaxis_title="Ship Type", yaxis_title="CO2 per NM", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Total Emissions by Type and Size (Aggregation)
    with col3:
        st.header("3. Total Emissions & Ship Size")
        df_agg = df_filtered.groupby('ShipType').agg(
            Total_CO2=('CO2_tonnes', 'sum'),
            Avg_DWT=('DWT', 'mean')
        ).reset_index()
        fig3 = px.bar(
            df_agg, x='ShipType', y='Total_CO2', title='Total Annual CO2 Emissions by Ship Type',
            color='Avg_DWT', color_continuous_scale=px.colors.sequential.Viridis,
            hover_data={'Avg_DWT': ':.0f'}
        )
        fig3.update_layout(xaxis_title="Ship Type", yaxis_title="Total CO2 (tonnes)", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: ML Prediction Accuracy (Scatter Plot)
    with col4:
        st.header("4. ML Model Validation (Actual vs. Predicted)")
        df_pred_chart = df_filtered.dropna(subset=['y_true_CO2', 'y_pred_CO2'])
        
        if len(df_pred_chart) > 0:
            st.info(f"Showing {len(df_pred_chart)} prediction points for selected Ship Type(s).")
            
            fig4 = px.scatter(
                df_pred_chart, x='y_true_CO2', y='y_pred_CO2',
                title='Actual vs. Predicted CO2 Emissions', hover_data=['ShipName', 'ShipType']
            )
            max_val = max(df_pred_chart['y_true_CO2'].max(), df_pred_chart['y_pred_CO2'].max())
            min_val = min(df_pred_chart['y_true_CO2'].min(), df_pred_chart['y_pred_CO2'].min())
            
            fig4.add_shape(type='line', line=dict(dash='dash', color='red'), x0=min_val, y0=min_val, x1=max_val, y1=max_val)
            fig4.update_layout(xaxis_title="Actual CO2 (tonnes)", yaxis_title="Predicted CO2 (tonnes)", height=400)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Prediction data is not available for the selected Ship Type(s). Please select a different combination of Ship Types.")
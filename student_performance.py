
# # Johaymen M. Malawani
# # ITD105 - IT4D.1

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats.mstats import winsorize

# Title of the app
st.title('Lab Exercise 1 - Exploratory Data Analysis')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    # 1. Load the data
    df = pd.read_csv(uploaded_file, delimiter=';')

    # 2. Sidebar Filters
    st.sidebar.header('Filters')
    
    # Filter by gender or sex
    if 'gender' in df.columns:
        gender_filter = st.sidebar.selectbox("Filter by Gender", ['All', 'Male', 'Female'])
        color_column = 'gender'
    elif 'sex' in df.columns:
        gender_filter = st.sidebar.selectbox("Filter by Sex", ['All', 'M', 'F'])
        color_column = 'sex'
    else:
        gender_filter = None
        color_column = None

    if gender_filter and gender_filter != 'All':
        df = df[df[color_column] == gender_filter]

    # Tabs for organizing sections
    tab1, tab2, tab3 = st.tabs(["Exploration", "Visualizations", "Insights & Analysis"])

    # 3. Basic Data Exploration
    with tab1:
        st.subheader('Data Preview')
        st.write(df.head())

        st.subheader('Summary Statistics')
        st.write(df.describe())

        st.subheader('Data Info')
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Missing Values Table
        st.subheader('Missing Values')
        missing_values = df.isnull().sum()
        st.write(pd.DataFrame(missing_values, columns=['Missing Values']))
    
    with tab2:
        st.subheader("Visualizations")

        # Visualization Type Selection
        viz_type = st.selectbox('Select Visualization Type', [
            'Default', 'Histograms', 'Density Plots', 'Box Plots', 'Bar Charts', 
            'Correlation Heatmap', 'Interactive Scatter Plot', 'Pair Plot'
        ])

        # Generate visualizations based on user selection
        if viz_type == 'Default':
            st.subheader('Summary of Important Visualizations')

            # Histograms
            st.markdown("### Histograms")
            col1, col2 = st.columns(2)
            for i, col in enumerate(df.select_dtypes(include=[np.number]).columns):
                with col1 if i % 2 == 0 else col2:
                    fig, ax = plt.subplots()
                    df[col].hist(ax=ax, bins=20)
                    ax.set_title(f'Histogram of {col}')
                    st.pyplot(fig)

            # Correlation Heatmap
            st.markdown("### Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 8})
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Box Plots
            st.markdown("### Box Plots")
            for col in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                st.pyplot(fig)

            # Scatter Plot for G3
            st.markdown("### Scatter Plot of Study Time vs Final Exam Score (G3)")
            fig = px.scatter(df, x="studytime", y="G3", color=color_column)
            st.plotly_chart(fig)

        elif viz_type == 'Histograms':
            st.subheader('Histograms')
            for col in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots()
                df[col].hist(ax=ax, bins=20)
                ax.set_title(f'Histogram of {col}')
                st.pyplot(fig)

        elif viz_type == 'Density Plots':
            st.subheader('Density Plots')
            for col in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots()
                sns.kdeplot(df[col], ax=ax, fill=True)
                ax.set_title(f'Density Plot of {col}')
                st.pyplot(fig)

        elif viz_type == 'Box Plots':
            st.subheader('Box Plots')
            for col in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                st.pyplot(fig)

        elif viz_type == 'Bar Charts':
            st.subheader('Bar Charts')
            for col in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots()
                if color_column:
                    df.groupby(color_column)[col].mean().plot(kind='bar', ax=ax)
                else:
                    df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Bar Chart of {col}')
                st.pyplot(fig)
                plt.close(fig)

        elif viz_type == 'Correlation Heatmap':
            st.subheader('Correlation Heatmap')
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 8})
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        elif viz_type == 'Interactive Scatter Plot':
            st.subheader('Interactive Scatter Plot')
            x_axis = st.selectbox('Select X-axis feature', df.select_dtypes(include=[np.number]).columns)
            y_axis = st.selectbox('Select Y-axis feature', ['G1', 'G2', 'G3'])
            scatter_plot = px.scatter(df, x=x_axis, y=y_axis, color=color_column)
            st.plotly_chart(scatter_plot)

        elif viz_type == 'Pair Plot':
            st.subheader('Pair Plot of Key Features')
            selected_features = st.multiselect(
                'Select features for the pair plot',
                df.select_dtypes(include=[np.number]).columns,
                default=['G1', 'G2', 'G3', 'studytime']
            )
            pair_plot = px.scatter_matrix(df, dimensions=selected_features, color=color_column)
            pair_plot.update_layout(height=800, width=900)
            st.plotly_chart(pair_plot)
    
        # Insights and Analysis Section
    with tab3:
        st.subheader('Outlier Detection using Z-Score')
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        outliers_zscore = (z_scores > 3).sum(axis=0)
        st.write('Number of outliers detected using Z-Score:')
        st.write(outliers_zscore)

        st.write('Removing Outliers...')
        df_no_outliers = df.copy()
        df_no_outliers = df_no_outliers[~((z_scores > 3).any(axis=1))]
        st.write('Data shape before removing outliers:', df.shape)
        st.write('Data shape after removing outliers:', df_no_outliers.shape)

        st.write('Applying log transformation...')
        df_transformed = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            df_transformed[col] = np.log1p(df_transformed[col])
        st.write('Data after log transformation:')
        st.write(df_transformed.head())

        st.write('Winsorizing...')
        df_winsorized = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            df_winsorized[col] = winsorize(df_winsorized[col], limits=[0.05, 0.05])
        st.write('Data after winsorizing:')
        st.write(df_winsorized.head())

        st.subheader('Insights from the Boxplot')
        st.write("Boxplots display the distribution of numerical features and reveal outliers and quartile ranges.")

        # Analysis on final exam score correlation
        st.subheader('Features with Highest Correlation to Final Exam Scores')
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        top_features_g1 = corr['G1'].sort_values(ascending=False).head(5)
        top_features_g2 = corr['G2'].sort_values(ascending=False).head(5)
        top_features_g3 = corr['G3'].sort_values(ascending=False).head(5)
        
        st.write("Top Features Correlated with G1 (First Period Grade):")
        st.write(top_features_g1)
        
        st.write("Top Features Correlated with G2 (Second Period Grade):")
        st.write(top_features_g2)
        
        st.write("Top Features Correlated with G3 (Final Exam Grade):")
        st.write(top_features_g3)

        # Correlation between study time and exam scores
        st.subheader('Study Time Correlation with Exam Scores')
        study_time_corr_g1 = df['studytime'].corr(df['G1'])
        study_time_corr_g2 = df['studytime'].corr(df['G2'])
        study_time_corr_g3 = df['studytime'].corr(df['G3'])
        st.write(f'Study Time vs G1 correlation: {study_time_corr_g1}')
        st.write(f'Study Time vs G2 correlation: {study_time_corr_g2}')
        st.write(f'Study Time vs G3 correlation: {study_time_corr_g3}')

        # Gender impact on final exam score
        st.subheader('Impact of Gender on Final Exam Score (G3)')
        if 'gender' in df.columns:
            gender_impact = df.groupby('gender')['G3'].mean()
        elif 'sex' in df.columns:
            gender_impact = df.groupby('sex')['G3'].mean()
        else:
            gender_impact = "No gender or sex column found."
        st.write('Average Final Exam Score (G3) by Gender:')
        st.write(gender_impact)

        # Boxplot insights
        st.subheader('Insights from the Boxplot')
        st.write("Boxplots show the spread and quartiles for numerical features and highlight potential outliers. This can reveal key distributions in the data, such as grades, study time, and other numeric attributes.")
        # (Optional: Add boxplot visualization for selected features)

        # Summary of Key Insights
        st.subheader('Key Insights Summary:')
        st.write("- The highest correlation with the final exam scores (G3) is observed in the second period grades (G2), followed by the first period grades (G1).")
        st.write("- Study time has a weak but positive correlation with exam performance, with the highest correlation seen with G1.")
        st.write("- Male students tend to have slightly higher average final exam scores than female students.")


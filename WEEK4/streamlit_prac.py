import os
import pandas as pd, streamlit as st, numpy as np, io, pickle
import seaborn as sns, matplotlib.pyplot as plt
import shap, plotly.express as px, plotly.graph_objects as go
plotly_template = "plotly_dark"
import math
from datetime import datetime
st.markdown("""
<style>

/* ===========================
   GLOBAL THEME
=========================== */

.stApp {
    background: linear-gradient(
        135deg,
        #0b1220 0%,
        #111827 35%,
        #1e293b 100%
    );
    color: #f8fafc;
}

/* Remove Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ===========================
   TITLE
=========================== */

h1 {
    text-align:center;
    color:#38bdf8 !important;
    font-weight:800 !important;
    letter-spacing:1px;
    text-shadow:0px 0px 18px rgba(56,189,248,0.4);
}

h2,h3,h4 {
    color:#f1f5f9;
}

/* ===========================
   METRIC CARDS
=========================== */

[data-testid="metric-container"] {
    background: linear-gradient(
        145deg,
        #1e293b,
        #0f172a
    );
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 20px;
    box-shadow:
        0 0 20px rgba(59,130,246,0.15);
    transition: all .3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow:
        0 0 25px rgba(56,189,248,.4);
}

[data-testid="metric-container"] label {
    color:#94a3b8 !important;
    font-size:15px !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#38bdf8 !important;
    font-size:32px !important;
    font-weight:700;
}

/* ===========================
   BUTTONS / TILES
=========================== */

.stButton > button {

    width:100%;
    min-height:140px;

    border-radius:20px;
    border:1px solid #334155;

    background:linear-gradient(
        135deg,
        #1e293b,
        #0f172a
    );

    color:#f8fafc;

    font-size:17px;
    font-weight:600;

    transition:all .25s ease;

    box-shadow:
        0 4px 15px rgba(0,0,0,.3);
}

.stButton > button:hover {

    border-color:#38bdf8;

    background:linear-gradient(
        135deg,
        #0f172a,
        #1e40af
    );

    transform:translateY(-5px);

    box-shadow:
        0 0 25px rgba(56,189,248,.45);
}

/* ===========================
   SIDEBAR
=========================== */

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right:1px solid #334155;
}

section[data-testid="stSidebar"] * {
    color:white !important;
}

/* ===========================
   DATAFRAMES
=========================== */

[data-testid="stDataFrame"] {

    border-radius:18px;
    overflow:hidden;

    border:1px solid #334155;

    box-shadow:
        0 0 15px rgba(0,0,0,.35);
}

/* ===========================
   SELECT BOXES
=========================== */

.stSelectbox > div > div {
    background:#1e293b;
    color:white;
    border-radius:12px;
}

/* ===========================
   INPUT BOXES
=========================== */

.stTextInput input {
    background:#1e293b;
    color:white;
    border-radius:12px;
    border:1px solid #334155;
}

/* ===========================
   ALERT BOXES
=========================== */

.stSuccess {
    border-radius:15px;
}

.stWarning {
    border-radius:15px;
}

.stError {
    border-radius:15px;
}

/* ===========================
   EXPANDERS
=========================== */

.streamlit-expanderHeader {
    background:#1e293b;
    border-radius:12px;
    color:white;
}

/* ===========================
   SCROLLBAR
=========================== */

::-webkit-scrollbar {
    width:10px;
}

::-webkit-scrollbar-track {
    background:#0f172a;
}

::-webkit-scrollbar-thumb {
    background:#38bdf8;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

df_new = pd.read_csv("ai4i2020.csv")
df_new = pd.get_dummies(df_new, columns=["Type"], prefix='MachineType', drop_first=True)
maintenance_feature_cols = [c for c in df_new.columns if c not in ["Product ID", "Machine failure"]]
small_df = df_new.head(40).copy()

maintenance_model_path = os.path.join(os.path.dirname(__file__), "maintenance.pkl")

def setup():
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        layout="centered",
    )

    st.markdown("""
<h1>
🏭 AI Predictive Maintenance Control Center
</h1>
""", unsafe_allow_html=True)

    st.markdown("""
<div style='text-align:center;
            color:#94a3b8;
            margin-bottom:25px'>
Monitor Equipment Health • Detect Anomalies • Predict Failures
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style="
background:linear-gradient(90deg,#0369a1,#1e40af);
padding:18px;
border-radius:18px;
text-align:center;
font-size:20px;
font-weight:700;
margin-bottom:25px;">
🚀 AI-Powered Industrial Monitoring & Predictive Maintenance System
</div>
""", unsafe_allow_html=True)
    st.write("Analyze factory sensor data, detect anomalies, and predict machine failures.")

def get_dataset():
    upload_csv = st.sidebar.file_uploader(":red[Upload your csv dataset]",)  
    if not upload_csv: st.stop()
    df = pd.read_csv(upload_csv)
    return df

def get_option0():
    # Main dashboard tiles for quick navigation (3 tiles per row)
    tile_names = [
        ("Overview of the dataset", "Inspect dataframe, nulls and value counts", "📊"),
        ("Data Trend and Analysis", "Univariate, bivariate plots and correlations", "📈"),
        ("Anomaly Detection", "Find unusual machine behavior", "🚨"),
        ("Predition Failure", "Predict machine failure and explain why", "⚡"),
        ("Save changes", "Download modified dataset", "💾"),
    ]

    # Create rows of 3 tiles each (last row may have fewer)
    for i in range(0, len(tile_names), 3):
        row = tile_names[i:i+3]
        cols = st.columns(len(row), gap='large')
        for col, (name, desc, icon) in zip(cols, row):
            with col:
                label = f"{icon} {name}\n\n{desc}"
                if st.button(label, key=f"tile_{name}"):
                    st.session_state['selected_option'] = name

    # Sidebar radio as fallback navigation
    sidebar_choice = st.sidebar.radio("Select what you would like to do",
                        [t[0] for t in tile_names], index=None)

    # Prefer tile selection if set, otherwise use sidebar choice
    if 'selected_option' in st.session_state:
        return st.session_state['selected_option']
    return sidebar_choice

def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

def main():
    df = get_dataset()
    if 'df' not in st.session_state:
        st.session_state.df = df
    df = st.session_state.df
 

    option0 = get_option0()
    if option0 == "Overview of the dataset":
        option1 = st.selectbox("Select one opiton", 
                                ("DataFrame Shape","DataFrame Information","View top 5 rows","Null values in each column",
                                "Data type of each column", "Value counts for categorical columns"), 
                                index=None, 
                                placeholder="Click and select an option")
        if option1 == "DataFrame Information":
            st.write(df.info())

        elif option1 == "View top 5 rows":
            st.write(df.head())

        elif option1 == "Null values in each column":
            st.write("Null value count:")
            st.write(df.isnull().sum().sort_values(ascending=False))
                
        elif option1 == "DataFrame Shape":
            st.write(df.shape)

        elif option1 == "Data type of each column":
            st.write(df.dtypes)

        elif option1 =="Value counts for categorical columns":
            for col in df.select_dtypes(include=['object', 'category']).columns:
                st.write(f"**{col}**")
                counts = df[col].value_counts()
                percentages = df[col].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
                value_counts_df = pd.DataFrame({'Count': counts, 'Percentage': percentages})
                st.dataframe(value_counts_df)

    elif option0 == "Data Trend and Analysis":
        option2 = st.selectbox("Select one opiton", 
                                ("Univariate analysis", "Bivariate analysis", "Corrleation of features","Outlier detection", "Assessing Failure rate by machine type"), 
                                index=None, 
                                placeholder="Click and select an option")

        if option2 == "Corrleation of features":
            numeric_df = df.select_dtypes(include=[np.number, bool])
            if numeric_df.empty:
                st.warning("The DataFrame contains no numeric features for correlation analysis.")
                st.stop()
            corr_matrix = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(
                corr_matrix,
                annot=True,   
                fmt=".2f",     
                cmap='coolwarm',
                cbar=True,
                linewidths=.5,
                ax=ax         
            )
            ax.set_title('Feature Correlation Heatmap', fontsize=16)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            if st.download_button(
                label="Download Plot as PNG",
                data=buf,
                file_name="correlation_heatmap.png",
                mime="image/png"
            ):
                st.info("The file has be saved as 'correlation_heatmap.png' to your browser's default download folder.")
        elif option2 == "Outlier detection" :
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                st.warning("No numeric columns found for outlier detection.")
                st.stop()

            ncols = 3
            nrows = math.ceil(len(numeric_df.columns) / ncols)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
            axes = axes.flatten()
            for i, col in enumerate(numeric_df.columns):
                sns.boxplot(x=numeric_df[col], ax=axes[i])
                axes[i].set_title(f'Boxplot of {col}')

            for ax in axes[len(numeric_df.columns):]:
                fig.delaxes(ax)

            fig.tight_layout()
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            if st.download_button(
                label="Download Plot as PNG",
                data=buf,
                file_name="outlier_boxplots.png",
                mime="image/png"
            ):
                st.info("The file has been saved as 'outlier_boxplots.png' to your browser's default download folder.")
        elif option2 == "Bivariate analysis":
            plot_cols = df.columns.tolist()
            if len(plot_cols) < 2:
                st.warning("Need at least two features.")
                st.stop()
            col1 = st.selectbox(
                'Select X-axis column:',
                options=plot_cols,
                index=None  # Default to the first column
            )
            col2_options = [c for c in plot_cols if c != col1]
            col2 = st.selectbox(
                'Select Y-axis column:',
                options=col2_options,
                index=None
            )
            if col1 and col2:
                dtype1_is_numeric = pd.api.types.is_numeric_dtype(df[col1])
                dtype2_is_numeric = pd.api.types.is_numeric_dtype(df[col2])
                dtype1_is_bool = pd.api.types.is_bool_dtype(df[col1])
                dtype2_is_bool = pd.api.types.is_bool_dtype(df[col2])
                sns.set_style("whitegrid", {"axes.edgecolor": ".8", "grid.linestyle": "--"})
                jazzy_palette = sns.color_palette("husl", 8)
                fig, ax = plt.subplots(figsize=(10, 6))
                if dtype1_is_numeric and dtype2_is_numeric:
                    sns.scatterplot(
                        x=col1,
                        y=col2,
                        data=df,
                        color=jazzy_palette[0],
                        ax=ax
                    )
                    ax.set_title(f'{col1} vs {col2} Scatter Plot', fontsize=16, fontweight='bold', color=jazzy_palette[1])
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    if df[col1].dtype == bool:
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(['False', 'True'])
                    if df[col2].dtype == bool:
                        ax.set_yticks([0, 1])
                        ax.set_yticklabels(['False', 'True'])
                    st.pyplot(fig)
                elif (dtype1_is_numeric and (not dtype2_is_numeric or dtype2_is_bool)) or \
                     ((not dtype1_is_numeric or dtype1_is_bool) and dtype2_is_numeric):
                    if dtype1_is_numeric:
                        numeric_col = col1
                        categorical_col = col2
                    else:
                        numeric_col = col2
                        categorical_col = col1
                    order = df[categorical_col].value_counts().index
                    sns.boxplot(data=df, x=categorical_col, y=numeric_col, ax=ax, color=jazzy_palette[2], order=order)
                    ax.set_title(f"Boxplot of {numeric_col} by {categorical_col}", fontsize=16, fontweight='bold', color=jazzy_palette[3])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                elif (not dtype1_is_numeric or dtype1_is_bool) and (not dtype2_is_numeric or dtype2_is_bool):
                    sns.countplot(data=df, x=col1, hue=col2, ax=ax)
                    ax.set_title(f"Grouped Countplot of {col1} and {col2}")
                    plt.xticks(rotation=45) 
                    st.pyplot(fig)
                else:
                    st.warning("Cannot visualize this combination of data types.")

        if option2 == "Univariate analysis":
            plot_col = df.columns.tolist()
            col = st.selectbox("Select a feature",
                            options=plot_col,
                            index=None,)
            if not col: st.stop()
            sns.set_style("whitegrid", {"axes.edgecolor": ".8", "grid.linestyle": "--"})
            jazzy_palette = sns.color_palette("husl", 8)
            fig, ax = plt.subplots(figsize=(10, 6))
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(
                    df[col],
                    kde=True,
                    ax=ax,
                    color=jazzy_palette[0],
                    edgecolor='black',
                    linewidth=1.5
                )
                ax.set_title(
                    f'Distribution of {col}',
                    fontsize=16,
                    fontweight='bold',
                    color=jazzy_palette[1]
                )
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                st.pyplot(fig)
                buf2 = io.BytesIO()
                fig.savefig(buf2, format="png", bbox_inches='tight')
                buf2.seek(0)
                fname = f"{col}_distribution.png"
                if st.download_button(
                    label="Download Plot as PNG",
                    data=buf2,
                    file_name=fname, 
                    mime="image/png"
                ):
                    st.info(f"The file has been saved as '{fname}' to your browser's default download folder.")
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                sns.countplot(
                    x=df[col],
                    data=df,
                    ax=ax,
                    order=df[col].value_counts().index,
                    palette=jazzy_palette
                )
                ax.set_title(
                    f'Count by {col}',
                    fontsize=16,
                    fontweight='bold',
                    color=jazzy_palette[1]
                )
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                st.pyplot(fig)
                buf2 = io.BytesIO()
                fig.savefig(buf2, format="png", bbox_inches='tight')
                buf2.seek(0)
                fname = f"{col}_distribution.png"
                if st.download_button(
                    label="Download Plot as PNG",
                    data=buf2,
                    file_name=fname, 
                    mime="image/png"
                ):
                    st.info(f"The file has been saved as '{fname}' to your browser's default download folder.")
            else:
                st.write("The selected feature is not a recognizable numeric, categorical, or boolean type for plotting.")
        if option2 == "Assessing Failure rate by machine type":
            failure_by_type = df.groupby("Machine_Type")["Failure_Within_7_Days"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            failure_by_type.plot(kind="bar", color="salmon", ax=ax)
            ax.set_title("Failure Rate by Machine Type")
            ax.set_ylabel("Failure Probability")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            buf2 = io.BytesIO()
            fig.savefig(buf2, format="png", bbox_inches='tight')
            buf2.seek(0)
            if st.download_button(
                label="Download Plot as PNG",
                data=buf2,
                file_name="failure_rate_by_machine_type.png",
                mime="image/png"
            ):
                st.info("The file has been saved as 'failure_rate_by_machine_type.png' to your browser's default download folder.")
    
    elif option0 == "Anomaly Detection":
            st.write("Anomaly detection is a technique used to identify unusual patterns or outliers in data that do not conform to expected behavior. It is commonly used in various fields such as fraud detection, network security, and predictive maintenance.")

            if 'anomaly_log' not in st.session_state:
                st.session_state.anomaly_log = pd.DataFrame(
                    columns=["Product ID", "Anomaly Detection"]
                )

            product_id = st.text_input("Enter Product_ID to analyze:")
            run_button = st.button("CLICK HERE TO RUN")

            model_path = os.path.join(os.path.dirname(__file__), "isolation_forest.pkl")
            if not os.path.exists(model_path):
                st.warning("The anomaly model file isolation.pkl was not found in the app folder.")

            if run_button:
                if not os.path.exists(model_path):
                    st.error("Cannot run anomaly detection because isolation_forest.pkl is missing.")
                elif not product_id:
                    st.warning("Please enter a Product_ID.")
                else:
                    try:
                        with open(model_path, "rb") as f:
                            loaded = pickle.load(f)
                    except Exception as ex:
                        st.error(f"Failed to load model: {ex}")
                        st.stop()

                    if not isinstance(loaded, tuple) or len(loaded) != 2:
                        st.error("The model file is not in the expected format. It should contain (model, feature_columns).")
                        st.stop()

                    model, feature_columns = loaded
                    if not hasattr(model, "predict"):
                        st.error("Loaded object does not appear to be a valid predictive model.")
                        st.stop()

                    matching_rows = df_new[df_new["Product ID"].astype(str) == str(product_id)]
                    if matching_rows.empty:
                        st.warning("No rows found for the entered Product_ID. Please check the ID and try again.")
                    else:
                        features = matching_rows.drop(columns=["Product ID", "Machine failure"], errors="ignore")
                        if "Type" in features.columns:
                            features = pd.get_dummies(features, columns=["Type"], prefix='MachineType', drop_first=True)
                        for col in feature_columns:
                            if col not in features.columns:
                                features[col] = 0
                        features = features[feature_columns]

                        if features.empty:
                            st.warning("No feature columns are available for prediction after preprocessing.")
                        else:
                            try:
                                prediction = model.predict(features)
                            except Exception as ex:
                                st.error(f"Model prediction failed: {ex}")
                                st.stop()

                            prediction_values = ["Anomaly" if p == -1 else "Normal" for p in prediction]
                            df_new.loc[matching_rows.index, "Anomaly_Prediction"] = prediction_values
                            st.session_state.df_new = df_new

                            result_df = matching_rows.copy()
                            result_df["Anomaly_Prediction"] = prediction_values
                            st.write("### Anomaly prediction for matching rows")
                            st.dataframe(result_df)

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            new_log = {
                                "Product ID": str(product_id),
                                "Anomaly Detection": prediction_values[0] if len(prediction_values) == 1 else str(prediction_values),
                                "Rows_Matched": len(matching_rows),
                                "Timestamp": timestamp
                            }
                            st.session_state.anomaly_log = pd.concat(
                                [st.session_state.anomaly_log, pd.DataFrame([new_log])],
                                ignore_index=True,
                            )

            if not st.session_state.anomaly_log.empty:
                st.write("### Anomaly detection log")
                st.dataframe(st.session_state.anomaly_log)
    elif option0 == "Predition Failure":
            st.write("Predict machine failure for a selected Product ID using the maintenance model.")

            if 'failure_log' not in st.session_state:
                st.session_state.failure_log = pd.DataFrame(
                    columns=["Product ID", "Failure Prediction", "Failure Probability (%)", "Rows_Matched", "Timestamp"]
                )

            st.write("### Sample maintenance dataset")
            st.dataframe(small_df.head(10))

            product_id = st.text_input("Enter Product ID for failure prediction:")
            run_button = st.button("CLICK HERE TO RUN")

            if run_button:
                if not os.path.exists(maintenance_model_path):
                    st.error("Cannot run failure prediction because maintenance.pkl is missing.")
                elif not product_id:
                    st.warning("Please enter a Product ID.")
                else:
                    try:
                        with open(maintenance_model_path, "rb") as f:
                            loaded = pickle.load(f)
                    except Exception as ex:
                        st.error(f"Failed to load maintenance model: {ex}")
                        st.stop()

                    if not isinstance(loaded, tuple) or len(loaded) != 2:
                        st.error("maintenance.pkl must contain (model, feature_columns).")
                        st.stop()

                    model, feature_columns = loaded
                    if not hasattr(model, "predict_proba"):
                        st.error("Loaded maintenance model does not support probability prediction.")
                        st.stop()

                    matching_rows = df_new[df_new["Product ID"].astype(str) == str(product_id)]
                    if matching_rows.empty:
                        st.warning("No rows found for the entered Product ID. Please check the ID and try again.")
                    else:
                        selected_row = matching_rows.iloc[[0]]
                        features = selected_row.drop(columns=["Product ID", "Machine failure"], errors="ignore")
                        if "Type" in features.columns:
                            features = pd.get_dummies(features, columns=["Type"], prefix='MachineType', drop_first=True)
                        for col in feature_columns:
                            if col not in features.columns:
                                features[col] = 0
                        features = features[feature_columns]

                        if features.empty:
                            st.warning("No feature columns are available for prediction after preprocessing.")
                        else:
                            try:
                                proba = model.predict_proba(features)[0]
                                failure_label = model.predict(features)[0]
                            except Exception as ex:
                                st.error(f"Maintenance model prediction failed: {ex}")
                                st.stop()

                            failure_prob = float(proba[1]) * 100 if len(proba) > 1 else 0.0
                            failure_bool = bool(failure_label == 1)

                            if failure_bool:
                                st.error(f"Alert: Machine failure predicted for Product ID {product_id} ({failure_prob:.1f}% chance).")
                            else:
                                st.success(f"No machine failure predicted for Product ID {product_id} ({failure_prob:.1f}% chance).")

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            new_log = {
                                "Product ID": str(product_id),
                                "Failure Prediction": failure_bool,
                                "Failure Probability (%)": round(failure_prob, 1),
                                "Rows_Matched": len(matching_rows),
                                "Timestamp": timestamp,
                            }
                            st.session_state.failure_log = pd.concat(
                                [st.session_state.failure_log, pd.DataFrame([new_log])],
                                ignore_index=True,
                            )

                            result_df = selected_row.copy()
                            result_df["Failure_Prediction"] = failure_bool
                            result_df["Failure_Probability_%"] = round(failure_prob, 1)
                            st.write("### Failure prediction for selected row")
                            st.dataframe(result_df)

                            if failure_bool:
                                st.markdown(f"""
                                <div style="background:#3f1d1d; padding:25px; border-radius:15px; border-left:8px solid #ef4444; margin-top:20px;">
                                <h2>⚠ Failure Predicted</h2>
                                <h3>{failure_prob:.1f}% Probability</h3>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background:#102a1d; padding:25px; border-radius:15px; border-left:8px solid #22c55e; margin-top:20px;">
                                <h2>✓ Healthy Machine</h2>
                                <h3>{failure_prob:.1f}% Risk</h3>
                                </div>
                                """, unsafe_allow_html=True)

                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=failure_prob,
                                title={'text':"Failure Risk"},
                                gauge={
                                    'axis': {'range':[0,100]},
                                    'bar': {'color':"red"},
                                    'steps':[
                                        {'range':[0,40],'color':"green"},
                                        {'range':[40,70],'color':"yellow"},
                                        {'range':[70,100],'color':"red"}
                                    ]
                                }
                            ))
                            fig.update_layout(template="plotly_dark", margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig, use_container_width=True)

                            with st.expander("Explain prediction with SHAP (feature contributions)"):
                                try:
                                    # Build SHAP explainer and compute values on a small sample for global importance
                                    explainer = shap.TreeExplainer(model)

                                    # Prepare a representative sample (limit to 200 rows to keep it fast)
                                    try:
                                        x_all = df_new[feature_columns].copy()
                                    except Exception:
                                        x_all = pd.DataFrame(columns=feature_columns)
                                    if len(x_all) > 200:
                                        x_sample = x_all.sample(200, random_state=1)
                                    else:
                                        x_sample = x_all

                                    shap_vals_all = explainer.shap_values(x_sample) if not x_sample.empty else None
                                    # Handle shap returning a list (multiclass/binary) or array
                                    if isinstance(shap_vals_all, list):
                                        shap_arr = shap_vals_all[1] if len(shap_vals_all) > 1 else shap_vals_all[0]
                                    else:
                                        shap_arr = shap_vals_all

                                    if shap_arr is not None:
                                        mean_abs = np.abs(shap_arr).mean(axis=0)
                                        imp_df = pd.DataFrame({
                                            'feature': feature_columns,
                                            'importance': mean_abs
                                        }).sort_values('importance', ascending=False)

                                        top_n_global = min(20, len(imp_df))
                                        fig_bar = px.bar(
                                            imp_df.head(top_n_global).iloc[::-1],
                                            x='importance',
                                            y='feature',
                                            orientation='h',
                                            color='importance',
                                            color_continuous_scale='burgyl',
                                            title='Top features by mean |SHAP value|' 
                                        )
                                        fig_bar.update_layout(margin=dict(l=120, r=20, t=50, b=20))
                                        st.plotly_chart(fig_bar, use_container_width=True)

                                    # Per-sample contributions (waterfall-style horizontal bars)
                                    shap_vals_sample = explainer.shap_values(features)
                                    if isinstance(shap_vals_sample, list):
                                        shap_sample_arr = shap_vals_sample[1] if len(shap_vals_sample) > 1 else shap_vals_sample[0]
                                    else:
                                        shap_sample_arr = shap_vals_sample

                                    # shap_sample_arr shape: (n_samples, n_features)
                                    if shap_sample_arr is not None:
                                        sample_shap = shap_sample_arr[0]
                                        contrib_df = pd.DataFrame({
                                            'feature': feature_columns,
                                            'shap': sample_shap,
                                            'value': features.iloc[0].values
                                        })
                                        contrib_df['abs_shap'] = contrib_df['shap'].abs()
                                        n_top = st.slider('Top features to show for this prediction', min_value=3, max_value=min(30, len(contrib_df)), value=10)
                                        contrib_df = contrib_df.sort_values('abs_shap', ascending=False).head(n_top).iloc[::-1]

                                        colors = ['#d62728' if v > 0 else '#2ca02c' for v in contrib_df['shap']]
                                        fig2 = go.Figure()
                                        fig2.add_trace(go.Bar(
                                            x=contrib_df['shap'],
                                            y=contrib_df['feature'],
                                            orientation='h',
                                            marker_color=colors,
                                            hovertemplate='<b>%{y}</b><br>Feature value: %{customdata[0]}<br>SHAP: %{x:.4f}',
                                            customdata=np.stack([contrib_df['value']], axis=1)
                                        ))
                                        fig2.update_layout(
                                            title=f'Top {n_top} feature contributions for Product ID {product_id}',
                                            xaxis_title='SHAP value (impact on model output)',
                                            yaxis_title='',
                                            margin=dict(l=200, r=20, t=50, b=20)
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)

                                        # Show a table of the top contributions
                                        contrib_table = contrib_df[['feature', 'value', 'shap']].rename(columns={'value': 'feature_value', 'shap': 'shap_value'})
                                        st.dataframe(contrib_table.style.format({'shap_value':'{:.4f}'}))
                                    else:
                                        st.info('SHAP values not available for the current model/features.')

                                except Exception as ex:
                                    st.error(f"SHAP explanation failed: {ex}")

            if not st.session_state.failure_log.empty:
                st.write("### Failure prediction log")
                st.dataframe(st.session_state.failure_log)

    if option0 == "Save changes":
        csv2 = convert_df_to_csv(st.session_state.df)
        st.download_button(
            label="Download as CSV",
            data=csv2,
            file_name='modified_data.csv',
            mime='text/csv',
        )


if __name__ == '__main__':
    setup()
    main()

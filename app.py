import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import re
import numpy as np
from textblob import TextBlob

# Page Config
st.set_page_config(page_title="Fragrance Emotional Lab", layout="wide", page_icon="🧪")

# --- NLP Engine ---
@st.cache_resource
def setup_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

def simple_clean(text):
    if not text or pd.isna(text): return []
    # Standardize to lowercase and remove non-alpha characters
    words = re.findall(r'\b[a-zà-ÿ]{3,}\b', str(text).lower())
    return [lemmatizer.lemmatize(w) for w in words]

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Data Upload")
    data_file = st.file_uploader("1. Upload Verbatim Excel", type=["xlsx"])
    dict_file = st.file_uploader("2. Upload Emotional Dictionary", type=["xlsx", "csv"])
    
    st.divider()
    st.subheader("🌍 Settings")
    dataset_lang = st.selectbox("Dataset Language:", ["English", "French", "German", "Spanish"])

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Emotional Load", "🌈 Fragrance Profiles", "📈 Competitive View"])

if data_file and dict_file:
    # Load Data
    df_raw = pd.read_excel(data_file)
    p_col = st.selectbox("Product ID Column", df_raw.columns)
    v_col = st.selectbox("Verbatim Column", df_raw.columns)

    # Load Dictionary
    if dict_file.name.endswith('.csv'):
        dict_df = pd.read_csv(dict_file)
    else:
        dict_df = pd.read_excel(dict_file)
    
    # Build word-to-emotion map
    emo_map = {}
    for _, row in dict_df.iterrows():
        cat = str(row.iloc[0]).strip() 
        sub = str(row.iloc[1]).strip() 
        word = str(row.iloc[2]).strip().lower() 
        if cat != "OUT" and word != "nan":
            emo_map[word] = {"cat": cat, "sub": sub}

    if st.sidebar.button("🚀 Analyze Emotional Impact"):
        df = df_raw.copy()
        def get_emotions(text):
            tokens = simple_clean(text)
            return [emo_map[t] for t in tokens if t in emo_map]

        df['matches'] = df[v_col].apply(get_emotions)
        df['has_emotion'] = df['matches'].apply(lambda x: 1 if len(x) > 0 else 0)
        # Force Product ID to string to prevent sorting errors later
        df[p_col] = df[p_col].astype(str)
        st.session_state['processed_emo'] = df

    if 'processed_emo' in st.session_state:
        df = st.session_state['processed_emo']
        
        # --- TAB 1: TOTAL EMOTIONAL LOAD ---
        with tab1:
            st.subheader("⚡ Total Emotional Load")
            # Calculate % of responses that triggered an emotion
            load_data = df.groupby(p_col)['has_emotion'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            load_data.sort_values().plot(kind='barh', color='#A2D2FF', ax=ax)
            ax.set_xlabel("% of Verbatims with Emotional Content")
            ax.set_ylabel("Fragrance Code")
            st.pyplot(fig)
            
            st.write("### Data Table")
            st.dataframe(load_data.rename("Emotional Load %").sort_values(ascending=False))

        # --- TAB 2: FRAGRANCE PROFILES ---
        with tab2:
            # FIX: Ensure unique product IDs are sorted as strings
            p_options = sorted(df[p_col].unique())
            target = st.selectbox("Select Fragrance to Inspect", p_options)
            
            sub_df = df[df[p_col] == target]
            all_matches = [item for sublist in sub_df['matches'] for item in sublist]
            
            if all_matches:
                m_df = pd.DataFrame(all_matches)
                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("**Main Category Split**")
                    cat_counts = m_df['cat'].value_counts(normalize=True) * 100
                    st.bar_chart(cat_counts)

                with c2:
                    st.write("**Sub-Dimension Profile**")
                    color_map = {"Emotion": "#FFADAD", "Image": "#A0C4FF", "Sensation": "#CAFFBF"}
                    sub_counts = m_df.groupby(['cat', 'sub']).size().reset_index(name='count')
                    sub_counts = sub_counts.sort_values('count', ascending=True)
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 8))
                    colors = [color_map.get(c, 'gray') for c in sub_counts['cat']]
                    ax2.barh(sub_counts['sub'], sub_counts['count'], color=colors)
                    st.pyplot(fig2)
            else:
                st.warning("No emotional keywords found in verbatims for this product.")

        # --- TAB 3: COMPETITIVE VIEW ---
        with tab3:
            st.subheader("⚔️ Competitive Mapping")
            all_emo_list = []
            for _, row in df.iterrows():
                for m in row['matches']:
                    all_emo_list.append({'pid': row[p_col], 'cat': m['cat']})
            
            if all_emo_list:
                comp_df = pd.DataFrame(all_emo_list)
                pivot_df = pd.crosstab(comp_df['pid'], comp_df['cat'], normalize='index') * 100
                
                st.write("Relative weight of Emotion vs Image vs Sensation")
                st.bar_chart(pivot_df)
                
                st.write("### Benchmark Table (%)")
                st.table(pivot_df.style.format("{:.1f}%").background_gradient(cmap="Blues"))
            else:
                st.info("Upload data and run analysis to see comparisons.")

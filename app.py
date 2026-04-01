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
    words = re.findall(r'\b[a-zà-ÿ]{3,}\b', str(text).lower())
    return [lemmatizer.lemmatize(w) for w in words]

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Data Upload")
    data_file = st.file_uploader("1. Upload Verbatim Excel", type=["xlsx"])
    dict_file = st.file_uploader("2. Upload Emotional Dictionary", type=["xlsx", "csv"])
    
    st.divider()
    st.subheader("🌍 Language Settings")
    dataset_lang = st.selectbox("Dataset Language:", ["English", "French", "German", "Spanish"])

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Emotional Load", "🌈 Fragrance Profiles", "📈 Competitive View"])

if data_file and dict_file:
    # Load Data
    df = pd.read_excel(data_file)
    p_col = st.selectbox("Product ID Column", df.columns)
    v_col = st.selectbox("Verbatim Column", df.columns)

    # Load & Process Dictionary
    if dict_file.name.endswith('.csv'):
        dict_df = pd.read_csv(dict_file)
    else:
        dict_df = pd.read_excel(dict_file)
    
    # Cleaning Dictionary: Mapping Category (A) -> Sub-Dimension (B) -> Keyword (C)
    # We create a mapping of keyword -> (Category, SubDimension)
    emo_map = {}
    for _, row in dict_df.iterrows():
        cat = str(row.iloc[0]).strip() # Column A
        sub = str(row.iloc[1]).strip() # Column B
        word = str(row.iloc[2]).strip().lower() # Column C
        if cat != "OUT":
            emo_map[word] = {"cat": cat, "sub": sub}

    if st.button("🚀 Analyze Emotional Impact"):
        # Process Verbatims
        def get_emotions(text):
            tokens = simple_clean(text)
            matches = [emo_map[t] for t in tokens if t in emo_map]
            return matches

        df['matches'] = df[v_col].apply(get_emotions)
        df['has_emotion'] = df['matches'].apply(lambda x: 1 if len(x) > 0 else 0)
        st.session_state['processed_emo'] = df

    if 'processed_emo' in st.session_state:
        df = st.session_state['processed_emo']
        
        # --- TAB 1: TOTAL EMOTIONAL LOAD ---
        with tab1:
            st.subheader("⚡ Total Emotional Load")
            load_data = df.groupby(p_col)['has_emotion'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 4))
            load_data.sort_values().plot(kind='barh', color='skyblue', ax=ax)
            ax.set_xlabel("% of Verbatims with Emotional Content")
            st.pyplot(fig)
            
            st.write("### Raw Load per Fragrance")
            st.dataframe(load_data.rename("Emotional Load %"))

        # --- TAB 2: FRAGRANCE PROFILES ---
        with tab2:
            target = st.selectbox("Select Fragrance to Inspect", sorted(df[p_col].unique()))
            sub_df = df[df[p_col] == target]
            
            # Flatten matches for this fragrance
            all_matches = [item for sublist in sub_df['matches'] for item in sublist]
            if all_matches:
                m_df = pd.DataFrame(all_matches)
                
                # 1. Main Category Distribution
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Main Category Split**")
                    cat_counts = m_df['cat'].value_counts(normalize=True) * 100
                    st.bar_chart(cat_counts)

                with c2:
                    st.write("**Sub-Dimension Profile**")
                    # Color Mapping
                    color_map = {"Emotion": "#ff9999", "Image": "#66b3ff", "Sensation": "#99ff99"}
                    sub_counts = m_df.groupby(['cat', 'sub']).size().reset_index(name='count')
                    sub_counts['color'] = sub_counts['cat'].map(color_map)
                    
                    fig2, ax2 = plt.subplots()
                    ax2.barh(sub_counts['sub'], sub_counts['count'], color=sub_counts['color'])
                    st.pyplot(fig2)
            else:
                st.warning("No emotional matches found for this fragrance.")

        # --- TAB 3: COMPETITIVE VIEW ---
        with tab3:
            st.subheader("⚔️ Comparative Emotional Profiles")
            
            # Create a pivot of Category % per Product
            all_emo_list = []
            for idx, row in df.iterrows():
                for m in row['matches']:
                    all_emo_list.append({'pid': row[p_col], 'cat': m['cat']})
            
            if all_emo_list:
                comp_df = pd.DataFrame(all_emo_list)
                pivot_df = pd.crosstab(comp_df['pid'], comp_df['cat'], normalize='index') * 100

                
                st.write("Relative weight of Emotion vs Image vs Sensation")
                st.bar_chart(pivot_df)
                
                st.write("### Benchmark Table (%)")
                st.table(pivot_df.style.background_gradient(cmap="YlGnBu"))

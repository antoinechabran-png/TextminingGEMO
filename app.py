import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import re
import numpy as np
from difflib import get_close_matches

# Page Config
st.set_page_config(page_title="Fragrance Emotional Lab Pro", layout="wide", page_icon="🧪")

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
    st.header("⚙️ Analysis Settings")
    data_file = st.file_uploader("1. Upload Verbatim Excel", type=["xlsx"])
    dict_file = st.file_uploader("2. Upload Emotional Dictionary", type=["xlsx", "csv"])
    st.divider()
    
    match_sensitivity = st.slider("Extrapolation Sensitivity", 0.6, 1.0, 1.0, 
                                  help="At 1.0, only Column C keywords are used. Below 1.0, the AI extrapolates using the synonyms in Column D.")
    dataset_lang = st.selectbox("Dataset Language:", ["English", "French", "German", "Spanish"])

tab1, tab2, tab3 = st.tabs(["📊 Emotional Load", "🌈 Fragrance Profiles", "📈 Competitive View"])

if data_file and dict_file:
    df_raw = pd.read_excel(data_file)
    p_col = st.selectbox("Product ID Column", df_raw.columns)
    v_col = st.selectbox("Verbatim Column", df_raw.columns)

    if dict_file.name.endswith('.csv'):
        dict_df = pd.read_csv(dict_file)
    else:
        dict_df = pd.read_excel(dict_file)
    
    emo_map = {}
    context_map = {}
    knowledge_pool = []

    for _, row in dict_df.iterrows():
        cat = str(row.iloc[0]).strip() 
        sub = str(row.iloc[1]).strip() 
        primary_word = str(row.iloc[2]).strip().lower()
        synonyms = str(row.iloc[3]).lower() if len(row) > 3 else ""

        if cat != "OUT" and primary_word != "nan" and primary_word != "":
            entry = {"cat": cat, "sub": sub}
            emo_map[primary_word] = entry
            knowledge_pool.append(primary_word)
            
            synonym_keywords = re.findall(r'\b[a-zà-ÿ]{3,}\b', synonyms)
            for kw in synonym_keywords:
                if kw not in emo_map:
                    context_map[kw] = entry
                    knowledge_pool.append(kw)

    knowledge_pool = list(set(knowledge_pool))

    if st.sidebar.button("🚀 Analyze Emotional Impact"):
        df = df_raw.copy()
        df = df.dropna(subset=[p_col, v_col])
        df[p_col] = df[p_col].astype(str).str.strip()
        
        def get_emotions(text):
            tokens = simple_clean(text)
            matches = []
            for t in tokens:
                if t in emo_map:
                    matches.append(emo_map[t])
                elif match_sensitivity < 1.0:
                    fuzzy_match = get_close_matches(t, knowledge_pool, n=1, cutoff=match_sensitivity)
                    if fuzzy_match:
                        res = emo_map.get(fuzzy_match[0]) or context_map.get(fuzzy_match[0])
                        if res: matches.append(res)
            return matches

        df['matches'] = df[v_col].apply(get_emotions)
        df['has_emotion'] = df['matches'].apply(lambda x: 1 if len(x) > 0 else 0)
        st.session_state['processed_emo'] = df

    if 'processed_emo' in st.session_state:
        df = st.session_state['processed_emo']
        
        with tab1:
            st.subheader("⚡ Total Emotional Load")
            load_data = (df.groupby(p_col)['has_emotion'].mean() * 100).sort_values()
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(load_data.index, load_data.values, color='#A2D2FF')
            ax.bar_label(bars, fmt='%.1f%%', padding=5)
            ax.set_xlabel("% of Verbatims with Emotional Content")
            st.pyplot(fig)
            st.dataframe(load_data.rename("Emotional Load %").sort_values(ascending=False))

        with tab2:
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
                    fig_cat, ax_cat = plt.subplots(figsize=(6, 6))
                    bars_cat = ax_cat.bar(cat_counts.index, cat_counts.values, color=['#FFADAD', '#A0C4FF', '#CAFFBF'])
                    ax_cat.bar_label(bars_cat, fmt='%.1f%%', padding=3)
                    ax_cat.set_ylabel("Share (%)")
                    st.pyplot(fig_cat)
                
                with c2:
                    st.write("**Sub-Dimension Profile**")
                    color_map = {"Emotion": "#FFADAD", "Image": "#A0C4FF", "Sensation": "#CAFFBF"}
                    sub_counts = m_df.groupby(['cat', 'sub']).size().reset_index(name='count').sort_values('count')
                    fig2, ax2 = plt.subplots(figsize=(8, 8))
                    colors = [color_map.get(c, 'gray') for c in sub_counts['cat']]
                    bars_sub = ax2.barh(sub_counts['sub'], sub_counts['count'], color=colors)
                    ax2.bar_label(bars_sub, padding=5)
                    ax2.set_xlabel("Mention Count")
                    st.pyplot(fig2)
            else:
                st.warning("No emotional matches detected.")

        with tab3:
            st.subheader("⚔️ Competitive Mapping")
            all_emo_list = [{'pid': row[p_col], 'cat': m['cat']} for _, row in df.iterrows() for m in row['matches']]
            if all_emo_list:
                comp_df = pd.DataFrame(all_emo_list)
                pivot_df = pd.crosstab(comp_df['pid'], comp_df['cat'], normalize='index') * 100
                
                # Plotting Competitive View with labels
                fig_comp, ax_comp = plt.subplots(figsize=(12, 7))
                pivot_df.plot(kind='bar', stacked=False, ax=ax_comp, color=['#FFADAD', '#A0C4FF', '#CAFFBF'])
                
                for container in ax_comp.containers:
                    ax_comp.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
                
                ax_comp.set_ylabel("Percentage (%)")
                ax_comp.set_xlabel("Product ID")
                plt.xticks(rotation=45)
                st.pyplot(fig_comp)
                
                st.table(pivot_df.style.format("{:.1f}%").background_gradient(cmap="Purples"))

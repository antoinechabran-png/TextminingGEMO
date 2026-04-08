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
    enable_crush = st.checkbox("❤️ Enable Crush Index")
    crush_dict_file = None
    crush_sheet = None
    if enable_crush:
        crush_dict_file = st.file_uploader("Upload Crush Dictionary", type=["xlsx"])
        if crush_dict_file:
            # Robust sheet detection to avoid ValueError
            try:
                xl = pd.ExcelFile(crush_dict_file)
                available_sheets = xl.sheet_names
                default_sheets = ["English", "French", "English For Translation"]
                # Filter defaults to what's actually in the file, or show all
                crush_sheet = st.selectbox("Select Crush Language Sheet", available_sheets)
            except Exception as e:
                st.error(f"Error reading Crush file: {e}")

    st.divider()
    match_sensitivity = st.slider("Extrapolation Sensitivity", 0.6, 1.0, 0.92)
    dataset_lang = st.selectbox("Dataset Language:", ["English", "French", "German", "Spanish"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Emotional Load", "🌈 Fragrance Profiles", "📈 Competitive View", "❤️ Crush Index"])

if data_file and dict_file:
    df_raw = pd.read_excel(data_file)
    p_col = st.selectbox("Product ID Column", df_raw.columns)
    v_col = st.selectbox("Verbatim Column", df_raw.columns)

    # Load Main Dictionary
    dict_df = pd.read_csv(dict_file) if dict_file.name.endswith('.csv') else pd.read_excel(dict_file)
    
    emo_map = {}
    context_map = {}
    knowledge_pool = []

    for _, row in dict_df.iterrows():
        cat, sub, primary_word = str(row.iloc[0]).strip(), str(row.iloc[1]).strip(), str(row.iloc[2]).strip().lower()
        synonyms = str(row.iloc[3]).lower() if len(row) > 3 else ""
        if cat != "OUT" and primary_word != "nan" and primary_word != "":
            entry = {"cat": cat, "sub": sub}
            emo_map[primary_word] = entry
            knowledge_pool.append(primary_word)
            for kw in re.findall(r'\b[a-zà-ÿ]{3,}\b', synonyms):
                if kw not in emo_map:
                    context_map[kw] = entry
                    knowledge_pool.append(kw)
    knowledge_pool = list(set(knowledge_pool))

    # Load Crush Dictionary
    crush_keywords = []
    if enable_crush and crush_dict_file and crush_sheet:
        try:
            crush_df = pd.read_excel(crush_dict_file, sheet_name=crush_sheet)
            crush_keywords = crush_df.iloc[:, 1].dropna().astype(str).str.lower().str.strip().tolist()
        except Exception as e:
            st.sidebar.error(f"Could not load Crush sheet: {e}")

    if st.sidebar.button("🚀 Analyze Emotional Impact"):
        df = df_raw.copy().dropna(subset=[p_col, v_col])
        df[p_col] = df[p_col].astype(str).str.strip()
        
        def get_emotions(text):
            tokens = simple_clean(text)
            matches = []
            i, negations = 0, ["not", "no", "pas", "non", "sans", "less", "peu", "un peu"]
            while i < len(tokens):
                trigram = " ".join(tokens[i:i+3]) if i < len(tokens) - 2 else None
                if trigram and trigram in emo_map:
                    matches.append(emo_map[trigram]); i += 3; continue
                bigram = " ".join(tokens[i:i+2]) if i < len(tokens) - 1 else None
                if bigram and bigram in emo_map:
                    matches.append(emo_map[bigram]); i += 2; continue
                if tokens[i] in negations:
                    i += 2; continue
                t = tokens[i]
                if t in emo_map:
                    matches.append(emo_map[t])
                elif match_sensitivity < 1.0:
                    fuzzy_match = get_close_matches(t, knowledge_pool, n=1, cutoff=match_sensitivity)
                    if fuzzy_match:
                        res = emo_map.get(fuzzy_match[0]) or context_map.get(fuzzy_match[0])
                        if res: matches.append(res)
                i += 1
            return matches

        def check_crush(text):
            if not crush_keywords: return 0
            text_lower = str(text).lower()
            return 1 if any(word in text_lower for word in crush_keywords) else 0

        df['matches'] = df[v_col].apply(get_emotions)
        df['has_emotion'] = df['matches'].apply(lambda x: 1 if len(x) > 0 else 0)
        if enable_crush:
            df['is_crush'] = df[v_col].apply(check_crush)
            
        st.session_state['processed_emo'] = df

    if 'processed_emo' in st.session_state:
        df = st.session_state['processed_emo']
        
        with tab1:
            st.subheader("⚡ Total Emotional Load")
            load_data = (df.groupby(p_col)['has_emotion'].mean() * 100).sort_values()
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(load_data.index, load_data.values, color='#A2D2FF')
            ax.bar_label(bars, fmt='%.1f%%', padding=5)
            st.pyplot(fig)

        with tab2:
            target = st.selectbox("Select Fragrance to Inspect", sorted(df[p_col].unique()))
            sub_df = df[df[p_col] == target]
            all_matches = [item for sublist in sub_df['matches'] for item in sublist]
            if all_matches:
                m_df = pd.DataFrame(all_matches)
                c1, c2 = st.columns(2)
                
                # --- COLOR MAPPING RE-ADDED HERE ---
                color_map = {"Emotion": "#FFADAD", "Image": "#A0C4FF", "Sensation": "#CAFFBF"}
                
                with c1:
                    st.write("**Main Category Split**")
                    cat_counts = m_df['cat'].value_counts(normalize=True) * 100
                    fig_cat, ax_cat = plt.subplots()
                    # Ensure colors match the category index
                    current_colors = [color_map.get(cat, "#D3D3D3") for cat in cat_counts.index]
                    ax_cat.bar(cat_counts.index, cat_counts.values, color=current_colors)
                    st.pyplot(fig_cat)
                with c2:
                    st.write("**Sub-Dimension Profile**")
                    sub_counts = m_df.groupby(['cat', 'sub']).size().reset_index(name='count').sort_values('count')
                    fig2, ax2 = plt.subplots()
                    # Apply specific colors to each bar based on its parent category
                    sub_colors = [color_map.get(c, 'gray') for c in sub_counts['cat']]
                    ax2.barh(sub_counts['sub'], sub_counts['count'], color=sub_colors)
                    st.pyplot(fig2)

        with tab3:
            st.subheader("⚔️ Competitive Mapping")
            all_emo_list = [{'pid': row[p_col], 'cat': m['cat']} for _, row in df.iterrows() for m in row['matches']]
            if all_emo_list:
                pivot_df = pd.crosstab(pd.DataFrame(all_emo_list)['pid'], pd.DataFrame(all_emo_list)['cat'], normalize='index') * 100
                st.bar_chart(pivot_df)
                st.table(pivot_df.style.format("{:.1f}%").background_gradient(cmap="Purples"))

        with tab4:
            st.subheader("❤️ Crush Index Analysis")
            if enable_crush and 'is_crush' in df.columns:
                crush_data = (df.groupby(p_col)['is_crush'].mean() * 100).sort_values()
                if not crush_data.empty:
                    fig_crush, ax_crush = plt.subplots(figsize=(10, 6))
                    bars_crush = ax_crush.barh(crush_data.index, crush_data.values, color='#FF6B6B')
                    ax_crush.bar_label(bars_crush, fmt='%.1f%%', padding=5)
                    ax_crush.set_xlabel("% of Verbatims expressing a 'Crush'")
                    st.pyplot(fig_crush)
                    st.dataframe(crush_data.rename("Crush Index %").sort_values(ascending=False))
                else:
                    st.warning("No data available for Crush Index. Check your dictionary and verbatims.")
            else:
                st.info("Please enable 'Crush Index' in the sidebar and upload the dictionary to see this analysis.")

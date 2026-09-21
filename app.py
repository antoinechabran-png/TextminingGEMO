import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import re
import numpy as np
import hashlib
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

    analysis_signature = (
        hashlib.sha256(data_file.getvalue()).hexdigest(),
        hashlib.sha256(dict_file.getvalue()).hexdigest(),
        p_col, v_col, enable_crush, crush_sheet, match_sensitivity, dataset_lang,
        hashlib.sha256(crush_dict_file.getvalue()).hexdigest() if crush_dict_file else None,
    )
    if st.session_state.get('analysis_signature') != analysis_signature:
        st.session_state.pop('processed_emo', None)

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

        def get_crush_matches(text):
            text_lower = str(text).lower()
            # Retain the original substring matching, excluding empty keywords.
            return list(dict.fromkeys(word for word in crush_keywords if word and word in text_lower))

        df['matches'] = df[v_col].apply(get_emotions)
        df['has_emotion'] = df['matches'].apply(lambda x: 1 if len(x) > 0 else 0)
        if enable_crush:
            df['crush_matches'] = df[v_col].apply(get_crush_matches)
            df['is_crush'] = df['crush_matches'].apply(lambda matches: int(bool(matches)))
            
        st.session_state['processed_emo'] = df
        st.session_state['analysis_signature'] = analysis_signature

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
                    plt.close(fig_crush)
                    summary_tab, verbatim_tab = st.tabs(["Index by product", "Crush verbatim extracts"])
                    with summary_tab:
                        st.dataframe(crush_data.rename("Crush Index %").sort_values(ascending=False))
                    with verbatim_tab:
                        selected_product = st.selectbox(
                            "Select fragrance / product code",
                            sorted(df[p_col].unique()),
                            key="crush_product_selector",
                        )
                        product_rows = df[df[p_col] == selected_product]
                        crush_rows = product_rows[product_rows['is_crush'] == 1]
                        st.caption(
                            f"{len(crush_rows)} crush verbatim(s) out of {len(product_rows)} "
                            f"analyzed verbatim(s) — Crush Index: "
                            f"{100 * len(crush_rows) / len(product_rows):.1f}%"
                        )
                        st.caption("Full original responses, with the dictionary terms that triggered each crush flag.")
                        if crush_rows.empty:
                            st.info("No crush verbatims were detected for this product.")
                        else:
                            extracts = pd.DataFrame({
                                "Product code": crush_rows[p_col].to_numpy(),
                                "Verbatim": crush_rows[v_col].to_numpy(),
                                "Matched crush terms": crush_rows['crush_matches'].apply(
                                    lambda terms: "; ".join(terms)
                                ).to_numpy(),
                            })
                            st.dataframe(extracts, hide_index=True, use_container_width=True)
                            st.download_button(
                                "Download selected product's crush verbatims (CSV)",
                                data=extracts.to_csv(index=False).encode("utf-8-sig"),
                                file_name="crush_verbatims.csv",
                                mime="text/csv",
                                key="download_crush_verbatims",
                            )
                else:
                    st.warning("No data available for Crush Index. Check your dictionary and verbatims.")
            else:
                st.info("Please enable 'Crush Index' in the sidebar and upload the dictionary to see this analysis.")
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import re
import numpy as np
import hashlib
from difflib import get_close_matches

# Page Config
st.set_page_config(page_title="Fragrance Emotional Lab Pro", layout="wide", page_icon="🧪")

NEGATORS = {
    'English': {'not', 'no', 'never', 'neither', 'nor', 'without', 'nothing', 'hardly', 'barely'},
    'French': {'ne', 'pas', 'jamais', 'aucun', 'aucune', 'sans', 'rien', 'ni', 'non'},
    'German': {'nicht', 'kein', 'keine', 'keinen', 'keinem', 'keiner', 'nie', 'ohne', 'nichts'},
    'Spanish': {'no', 'nunca', 'sin', 'ningún', 'ninguna', 'ninguno', 'nada', 'ni', 'tampoco'},
}
CONTRASTS = {'but', 'however', 'yet', 'mais', 'pourtant', 'aber', 'jedoch', 'pero', 'sino', 'aunque'}
NEGATIVE_PREFIXES = {
    'English': ('un', 'dis', 'non', 'in', 'im', 'ir', 'il'),
    'French': ('dé', 'dés', 'in', 'im', 'ir', 'il', 'mal', 'non'),
    'German': ('un', 'miss'),
    'Spanish': ('des', 'in', 'im', 'ir', 'il'),
}


def tokenize(text):
    text = str(text).casefold().replace('’', "'")
    text = re.sub(r"\bcan't\b", 'can not', text)
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"\bcannot\b", 'can not', text)
    text = re.sub(r"n['’]t\b", ' not', text)
    text = re.sub(r"\bn'(?=\w)", 'ne ', text)
    return re.findall(r"[^\W\d_]+|[.,;:!?\n]", text, flags=re.UNICODE)


class DictionaryMatcher:
    """Conservative, reviewable dictionary matching; not a sentiment model."""

    def __init__(self, entries, language, sensitivity=1.0):
        self.language = language
        self.stemmer = SnowballStemmer(language.lower())
        self.sensitivity = sensitivity
        self.entries = {}
        self.stems = {}
        for term, entry in entries.items():
            tokens = tuple(tokenize(term))
            if not tokens:
                continue
            self.entries[tokens] = (term, entry)
            self.stems.setdefault(tuple(self.stemmer.stem(t) for t in tokens), (term, entry))
        self.max_words = max((len(t) for t in self.entries), default=1)
        self.single_words = [t[0] for t in self.entries if len(t) == 1]

    def negative_affix(self, word):
        # Never strip these to obtain a positive match, even through fuzzy matching.
        for prefix in NEGATIVE_PREFIXES[self.language]:
            if word.startswith(prefix) and len(word) - len(prefix) >= 3:
                root = word[len(prefix):]
                if (root,) in self.entries or (self.stemmer.stem(root),) in self.stems:
                    return True
        if self.language == 'English' and word.endswith('less'):
            return True
        if self.language == 'German' and word.endswith('los'):
            return True
        return False

    def scan(self, text):
        tokens = tokenize(text)
        hits = []
        negated = False
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in '.,;:!?\n' or token in CONTRASTS:
                negated = False
                i += 1
                continue
            if token in NEGATORS[self.language]:
                # 'not only fresh' is not a denial of freshness.
                exception = tokens[i:i+2] in (['not', 'only'], ['pas', 'seulement'], ['nicht', 'nur'], ['no', 'solo'])
                if not exception:
                    negated = True
                i += 2 if exception else 1
                continue
            found = None
            for size in range(min(self.max_words, len(tokens) - i), 0, -1):
                span = tuple(tokens[i:i+size])
                if any(t in NEGATORS[self.language] or t in CONTRASTS or t in '.,;:!?\n' for t in span):
                    continue
                if span in self.entries:
                    found = (*self.entries[span], 'exact', size)
                    break
                # Probability words must not stem/fuzzy-match the preference 'like'.
                if self.language == 'English' and any(t in {'likely', 'unlikely', 'likelihood'} for t in span):
                    continue
                if any(self.negative_affix(t) for t in span):
                    continue
                stem = tuple(self.stemmer.stem(t) for t in span)
                if stem in self.stems:
                    found = (*self.stems[stem], 'word form', size)
                    break
            if found is None and self.sensitivity < 1 and len(token) >= 4 and not self.negative_affix(token) and token not in {'likely', 'unlikely', 'likelihood'}:
                candidates = [w for w in self.single_words if not self.negative_affix(w)]
                close = get_close_matches(token, candidates, n=1, cutoff=self.sensitivity)
                if close:
                    found = (*self.entries[(close[0],)], 'fuzzy', 1)
            if found:
                term, entry, method, size = found
                # Also recognize common German/French postposed negation.
                after = tokens[i+size:i+size+2]
                post_negated = self.language in {'French', 'German'} and any(t in {'pas', 'nicht'} for t in after)
                hits.append({**entry, 'term': term, 'matched_text': ' '.join(tokens[i:i+size]),
                             'method': method, 'negated': negated or post_negated})
                i += size
            else:
                i += 1
        return hits


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

    analysis_signature = (
        "negation-word-forms-v2",
        hashlib.sha256(data_file.getvalue()).hexdigest(),
        hashlib.sha256(dict_file.getvalue()).hexdigest(),
        p_col, v_col, enable_crush, crush_sheet, match_sensitivity, dataset_lang,
        hashlib.sha256(crush_dict_file.getvalue()).hexdigest() if crush_dict_file else None,
    )
    if st.session_state.get('analysis_signature') != analysis_signature:
        st.session_state.pop('processed_emo', None)

    if st.sidebar.button("🚀 Analyze Emotional Impact"):
        df = df_raw.copy().dropna(subset=[p_col, v_col])
        df[p_col] = df[p_col].astype(str).str.strip()
        
        emotional_matcher = DictionaryMatcher({**context_map, **emo_map}, dataset_lang, match_sensitivity)
        df['emotion_evidence'] = df[v_col].apply(emotional_matcher.scan)
        df['matches'] = df['emotion_evidence'].apply(lambda hits: [h for h in hits if not h['negated']])
        df['has_emotion'] = df['matches'].apply(lambda hits: int(bool(hits)))
        if enable_crush:
            crush_matcher = DictionaryMatcher({term: {} for term in crush_keywords}, dataset_lang)
            df['crush_evidence'] = df[v_col].apply(crush_matcher.scan)
            df['crush_matches'] = df['crush_evidence'].apply(
                lambda hits: list(dict.fromkeys(h['term'] for h in hits if not h['negated']))
            )
            df['is_crush'] = df['crush_matches'].apply(lambda hits: int(bool(hits)))

        st.session_state['processed_emo'] = df
        st.session_state['analysis_signature'] = analysis_signature

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
            st.caption(f"{len(sub_df)} analyzed verbatim(s). Category percentages use detected mentions, not respondents.")
            with st.expander("Review emotional matches and excluded negations"):
                evidence = [
                    {"Verbatim": row[v_col], "Dictionary term": h['term'],
                     "Matched text": h['matched_text'], "Match type": h['method'],
                     "Status": "Excluded: negated" if h['negated'] else "Counted",
                     "Category": h['cat'], "Sub-dimension": h['sub']}
                    for _, row in sub_df.iterrows() for h in row['emotion_evidence']
                ]
                if evidence:
                    st.dataframe(pd.DataFrame(evidence), hide_index=True, use_container_width=True)
                else:
                    st.info("No dictionary matches found.")
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
            st.caption("Percentages are shares of detected emotion mentions, not respondents.")
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
                    plt.close(fig_crush)
                    with st.expander("Review excluded negated crush matches"):
                        excluded = [
                            {"Product code": row[p_col], "Verbatim": row[v_col],
                             "Dictionary term": h['term'], "Matched text": h['matched_text']}
                            for _, row in df.iterrows() for h in row['crush_evidence'] if h['negated']
                        ]
                        if excluded:
                            st.dataframe(pd.DataFrame(excluded), hide_index=True, use_container_width=True)
                        else:
                            st.info("No negated crush matches found.")
                    summary_tab, verbatim_tab = st.tabs(["Index by product", "Crush verbatim extracts"])
                    with summary_tab:
                        st.dataframe(crush_data.rename("Crush Index %").sort_values(ascending=False))
                    with verbatim_tab:
                        selected_product = st.selectbox(
                            "Select fragrance / product code",
                            sorted(df[p_col].unique()),
                            key="crush_product_selector",
                        )
                        product_rows = df[df[p_col] == selected_product]
                        crush_rows = product_rows[product_rows['is_crush'] == 1]
                        st.caption(
                            f"{len(crush_rows)} crush verbatim(s) out of {len(product_rows)} "
                            f"analyzed verbatim(s) — Crush Index: "
                            f"{100 * len(crush_rows) / len(product_rows):.1f}%"
                        )
                        st.caption("Full original responses, with the dictionary terms that triggered each crush flag.")
                        if crush_rows.empty:
                            st.info("No crush verbatims were detected for this product.")
                        else:
                            extracts = pd.DataFrame({
                                "Product code": crush_rows[p_col].to_numpy(),
                                "Verbatim": crush_rows[v_col].to_numpy(),
                                "Matched crush terms": crush_rows['crush_matches'].apply(
                                    lambda terms: "; ".join(terms)
                                ).to_numpy(),
                            })
                            st.dataframe(extracts, hide_index=True, use_container_width=True)
                            st.download_button(
                                "Download selected product's crush verbatims (CSV)",
                                data=extracts.to_csv(index=False).encode("utf-8-sig"),
                                file_name="crush_verbatims.csv",
                                mime="text/csv",
                                key="download_crush_verbatims",
                            )
                else:
                    st.warning("No data available for Crush Index. Check your dictionary and verbatims.")
            else:
                st.info("Please enable 'Crush Index' in the sidebar and upload the dictionary to see this analysis.")

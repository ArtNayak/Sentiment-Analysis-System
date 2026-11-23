import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

from utils import extract_key_themes, filter_spam, is_safe_url

MAX_COMMENTS = 500  # keeps one analysis run to a sensible length on CPU
SUMMARY_CHUNK_WORDS = 500  # DistilBART reads at most 1024 tokens, so long text is split

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Comment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling (No changes here) ---
css_string = """
/* Card styling for the dashboard */
.dashboard-card { padding: 25px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s ease; backdrop-filter: blur(10px); height: 100%; }
.dashboard-card:hover { transform: translateY(-5px); border: 1px solid rgba(147, 112, 219, 0.5); }
.dashboard-card h3 { font-size: 1.5rem; color: #FFFFFF; }
.dashboard-card p { color: #B0B0B0; }
"""
st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)


# --- OPTIMIZATION: Model Loading is now split into separate functions ---

@st.cache_resource
def load_roberta_model():
    """Loads only the RoBERTa model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    return tokenizer, model

@st.cache_resource
def load_summarizer():
    """Loads only the summarization pipeline."""
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device="cpu")
    return summarizer

# --- Analysis Functions (Updated to accept models as arguments) ---
def analyze_roberta_sentiment(text, tokenizer, model):
    processed_text = " ".join([t if not t.startswith('@') else '@user' for t in text.split()])
    encoded_text = tokenizer(processed_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_text)
    scores = F.softmax(output.logits, dim=1)[0].tolist()
    return {'Negative': scores[0], 'Neutral': scores[1], 'Positive': scores[2]}

def generate_summary(text, min_len, max_len, summarizer):
    words = text.split()
    if len(words) <= 40:
        return "The provided text is too short to generate a meaningful summary."
    options = dict(max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
    chunks = [" ".join(words[i:i + SUMMARY_CHUNK_WORDS]) for i in range(0, len(words), SUMMARY_CHUNK_WORDS)]
    if len(chunks) > 1 and len(chunks[-1].split()) < 40:  # fold a tiny tail into the previous chunk
        chunks[-2] += " " + chunks.pop()
    parts = [summarizer(chunk, **options)[0]['summary_text'] for chunk in chunks]
    summary = " ".join(parts)
    if len(parts) > 1:  # second pass so the final summary respects the requested length
        summary = summarizer(summary, **options)[0]['summary_text']
    return summary

# --- Other functions (No changes here) ---
def create_word_cloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA", colormap='viridis').generate(text)
    except ValueError:  # nothing left to draw after WordCloud removes stop words
        return None
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def scrape_comments(url):
    safe, reason = is_safe_url(url)
    if not safe:
        st.error(reason)
        return []
    comments = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until='networkidle', timeout=20000)
                # best effort: a redirect must not land on a private address either
                if not is_safe_url(page.url)[0]:
                    st.error("The page redirected to a non-public address, so it was not read.")
                    return []
                for element in page.query_selector_all("p"):
                    text = element.inner_text()
                    if text and len(text.split()) > 8:
                        comments.append(text)
            finally:
                browser.close()
        if not comments:
            st.warning("Could not find significant text using the generic 'p' tag selector.")
        return comments
    except PlaywrightTimeoutError:
        st.error(f"Timeout Error: The page at {url} took too long to load.")
    except Exception as e:
        st.error(f"An error occurred during scraping: {e}")
    return []

# --- UI Rendering Functions ---

def render_dashboard():
    st.title("AI-Powered E-Consultation Suite")
    st.markdown("##### Leverage AI to streamline stakeholder feedback review. Select a tool from the sidebar or a card below to begin.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown(
                """<div class="dashboard-card"><h3>🧠 Sentiment Analysis</h3><p>Analyze comments from text or a URL to gauge public opinion. Get detailed breakdowns, key themes, and more.</p></div>""", unsafe_allow_html=True)
            if st.button("Launch Sentiment Analysis", use_container_width=True, key="nav_senti"):
                st.session_state.page = "Sentiment Analysis"
                st.rerun()
    with col2:
        with st.container():
            st.markdown(
                """<div class="dashboard-card"><h3>📝 Smart Summaries</h3><p>Paste long documents or articles to generate concise, AI-powered summaries. Perfect for quick insights.</p></div>""", unsafe_allow_html=True)
            if st.button("Launch Smart Summarizer", use_container_width=True, key="nav_summary"):
                st.session_state.page = "Smart Summaries"
                st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Coming Soon: Trend Visualization and Dynamic Word Clouds", icon="✨")

def render_analysis_page():
    st.header("🧠 Sentiment Analysis")
    
    # --- OPTIMIZATION: Load model only when this page is opened ---
    with st.spinner("Loading Sentiment Analysis model..."):
        roberta_tokenizer, roberta_model = load_roberta_model()
    
    st.markdown("Provide text to analyze sentiment, extract key themes, and generate insights.")
    input_type = st.radio("Choose input method:", ("Direct Text Input", "Scrape from URL"), horizontal=True)
    comments_to_analyze = []
    if input_type == "Direct Text Input":
        user_input = st.text_area("Paste comments here:", height=200, placeholder="This is a fantastic initiative.\nThe proposal has some major flaws.")
        if user_input: comments_to_analyze = [c.strip() for c in user_input.split('\n') if c.strip()]
    else:
        url = st.text_input("Enter the URL to scrape comments from:")
        if st.button("Scrape Comments"):
            with st.spinner(f"Scraping {url}..."):
                scraped_comments = scrape_comments(url)
                if scraped_comments:
                    st.session_state.scraped_text = "\n".join(scraped_comments)
                    st.success(f"Successfully scraped {len(scraped_comments)} paragraphs.")
        user_input = st.text_area("Scraped/Editable Text:", value=st.session_state.get('scraped_text', ''), height=200)
        if user_input: comments_to_analyze = [c.strip() for c in user_input.split('\n') if c.strip()]

    filter_on = st.checkbox("Remove spam and duplicate comments before analysis", value=True)

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        removed = 0
        if filter_on:
            comments_to_analyze, removed = filter_spam(comments_to_analyze)
        if len(comments_to_analyze) > MAX_COMMENTS:
            st.warning(f"Only the first {MAX_COMMENTS} comments are analyzed.")
            comments_to_analyze = comments_to_analyze[:MAX_COMMENTS]
        if removed:
            st.info(f"Removed {removed} spam or duplicate comment(s).")
        if not comments_to_analyze:
            st.warning("Please provide text to analyze.")
        else:
            with st.spinner("AI is thinking..."):
                all_roberta_scores = [analyze_roberta_sentiment(c, roberta_tokenizer, roberta_model) for c in comments_to_analyze]
                avg_scores = pd.DataFrame(all_roberta_scores).mean()
                st.session_state.analysis_results = { 'overall_sentiment': avg_scores.idxmax(), 'confidence': avg_scores.max(), 'avg_scores': avg_scores, 'full_text': " ".join(comments_to_analyze), 'num_comments': len(comments_to_analyze) }
    
    if 'analysis_results' in st.session_state:
        # (The rest of the display logic is the same)
        results = st.session_state.analysis_results
        st.markdown("---"); st.subheader("📊 Analysis Results")
        sentiment_color = "green" if results['overall_sentiment'] == 'Positive' else "red" if results['overall_sentiment'] == 'Negative' else "orange"
        st.markdown(f"### Overall Sentiment: <span style='color:{sentiment_color};'>{results['overall_sentiment']}</span> (Confidence: {results['confidence']:.1%})", unsafe_allow_html=True)
        col1, col2 = st.columns(2);
        with col1:
            st.write("**Detailed Breakdown**")
            fig = px.bar(x=results['avg_scores'].index, y=results['avg_scores'].values, labels={'x': 'Sentiment', 'y': 'Score'}, color=results['avg_scores'].index, color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'}, text=[f"{v:.1%}" for v in results['avg_scores'].values])
            fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            st.write("**🔑 Key Themes**"); key_themes = extract_key_themes(results['full_text'])
            if not key_themes:
                st.caption("No key themes found.")
            for theme in key_themes: st.markdown(f"- `{theme.title()}`")
        with col2:
            st.write("**☁️ Word Cloud**"); wordcloud_fig = create_word_cloud(results['full_text'])
            if wordcloud_fig is None:
                st.caption("Not enough words to draw a word cloud.")
            else:
                st.pyplot(wordcloud_fig)
                plt.close(wordcloud_fig)

def render_summary_page():
    st.header("📝 Smart Summarizer")
    
    # --- OPTIMIZATION: Load model only when this page is opened ---
    with st.spinner("Loading Summarization model..."):
        summarizer = load_summarizer()

    st.markdown("Paste a long article, report, or any block of text to generate a concise summary.")
    source_text = st.text_area("Paste your text here:", height=300, placeholder="Paste a long article here...")
    st.markdown("### Configure Summary Length")
    col1, col2 = st.columns(2)
    with col1: min_len = st.slider("Minimum summary length (tokens)", 20, 200, 30)
    with col2: max_len = st.slider("Maximum summary length (tokens)", 50, 500, 150)

    if st.button("Generate Summary", type="primary", use_container_width=True):
        if min_len >= max_len:
            st.warning("The minimum length must be smaller than the maximum length.")
        elif len(source_text.split()) > 40:
            with st.spinner("Generating summary..."):
                summary = generate_summary(source_text, min_len, max_len, summarizer)
                st.session_state.summary_output = { 'summary': summary, 'original_wc': len(source_text.split()), 'summary_wc': len(summary.split()) }
        else:
            st.warning("Please enter text with more than 40 words.")
    
    if 'summary_output' in st.session_state:
        results = st.session_state.summary_output
        st.markdown("---"); st.subheader("📄 Generated Summary"); st.success(results['summary'])
        reduction = 100 - (results['summary_wc'] / results['original_wc'] * 100)
        st.info(f"**Original Word Count:** {results['original_wc']} | **Summary Word Count:** {results['summary_wc']} | **Reduction:** {reduction:.1f}%")

# --- Main App Router (No changes here) ---
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

pages = ["Dashboard", "Sentiment Analysis", "Smart Summaries"]
icons = ['house-door-fill', 'bar-chart-line-fill', 'file-text-fill']
try:
    current_page_index = pages.index(st.session_state.page)
except ValueError:
    current_page_index = 0

with st.sidebar:
    selected_page = option_menu("Main Menu", pages, icons=icons, menu_icon="cast", default_index=current_page_index, styles={ "container": {"background-color": "#1a1a1a"}, "icon": {"color": "#9370DB", "font-size": "20px"}, "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#4B0082"}, "nav-link-selected": {"background-color": "#4B0082"}, })
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

if st.session_state.page == "Dashboard":
    render_dashboard()
elif st.session_state.page == "Sentiment Analysis":
    render_analysis_page()
elif st.session_state.page == "Smart Summaries":
    render_summary_page()
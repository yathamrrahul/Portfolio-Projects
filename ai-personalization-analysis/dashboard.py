import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="AI Personalization Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Netflix-inspired color palette
COLORS = {
    'primary': '#E50914',      # Netflix red
    'secondary': '#221F1F',    # Dark background
    'accent': '#B20710',       # Darker red
    'success': '#46D369',      # Green
    'warning': '#F5A623',      # Orange/Gold
    'error': '#E50914',        # Red
    'text': '#FFFFFF',         # White
    'text_secondary': '#B3B3B3', # Gray
    'bg_card': '#2F2F2F',      # Card background
    'bg_dark': '#141414',      # Dark background
}

# Custom CSS
st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: {COLORS['bg_dark']};
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['secondary']};
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {COLORS['text']};
    }}
    
    /* Headers */
    .main-header {{
        font-size: 3rem;
        font-weight: 700;
        color: {COLORS['text']};
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -1px;
    }}
    
    .section-header {{
        font-size: 2rem;
        font-weight: 600;
        color: {COLORS['text']};
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid {COLORS['primary']};
        padding-bottom: 0.5rem;
    }}
    
    /* Metric cards - equal width */
    div[data-testid="stMetric"] {{
        background-color: {COLORS['bg_card']};
        padding: 1rem;
        border-radius: 8px;
        min-width: 200px;
    }}
    
    /* Info boxes */
    .stAlert {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text']};
        border-radius: 8px;
    }}
    
    /* Text color overrides */
    .stMarkdown, .stText, p, span, div {{
        color: {COLORS['text']} !important;
    }}
    
    /* Dataframe styling */
    .dataframe {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text']};
    }}
    
    /* Download buttons */
    .stDownloadButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }}
    
    .stDownloadButton > button:hover {{
        background-color: {COLORS['accent']};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    [
        "Executive Summary",
        "Experiment 1: Baselines",
        "Experiment 2: Features",
        "Conditional Analysis",
        "Business Recommendations"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Analysis")
st.sidebar.info(
    "**AI Personalization Study**\n\n"
    "Rigorous comparison of AI models versus simple baselines "
    "for recommendation systems.\n\n"
    "**Dataset:** MovieLens 1M\n"
    "**Scale:** 1M ratings, 6K users, 4K movies\n\n"
    "**Author:** Rahul Yatham"
)

# Quick stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Control Performance", "42.0%", "+28% vs AI")
st.sidebar.metric("Models Evaluated", "5")
st.sidebar.metric("Conditions Tested", "13")

# ============================================================
# DATA LOADING & UTILITY FUNCTIONS
# ============================================================

@st.cache_data
def load_results():
    """Load pre-computed experimental results"""
    results = {
        'overall': pd.DataFrame({
            'Model': ['Control', 'SVD Pure', 'SVD Hybrid', 'KNN Pure', 'KNN Hybrid'],
            'HitRate@10': [0.420, 0.263, 0.302, 0.269, 0.312],
            'MRR@10': [0.187, 0.112, 0.106, 0.098, 0.104],
            'Training Time (s)': [0, 45, 45, 382, 382],
            'Inference Time (s)': [0.001, 28, 28, 1114, 1114]
        }),
        'segments': pd.DataFrame({
            'Segment': ['Casual\n(49 ratings)', 'Active\n(105 ratings)', 
                       'Heavy\n(351 ratings)', 'Minimal\n(29 ratings)'],
            'Control': [0.269, 0.521, 0.750, 0.278],
            'AI Best': [0.224, 0.367, 0.562, 0.236]
        }),
        'temporal': pd.DataFrame({
            'Context': ['Weekday', 'Weekend', 'Winter', 'Spring', 'Summer', 'Fall'],
            'Control': [0.429, 0.471, 0.502, 0.499, 0.439, 0.421],
            'AI Best': [0.320, 0.355, 0.360, 0.370, 0.333, 0.320]
        }),
        'feature_impact': pd.DataFrame({
            'Model': ['SVD', 'KNN'],
            'Pure': [0.263, 0.269],
            'Hybrid': [0.302, 0.312],
            'Improvement': ['+15%', '+16%']
        })
    }
    return results

def create_download_button(fig, filename):
    """Create download button for plotly figures"""
    buffer = io.BytesIO()
    fig.write_image(buffer, format='png', width=1200, height=600, scale=2)
    buffer.seek(0)
    
    st.download_button(
        label="Download Chart",
        data=buffer,
        file_name=filename,
        mime="image/png"
    )

results = load_results()
# ============================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":
    
    # Hero section
    st.markdown(
        '<h1 class="main-header">AI Personalization Analysis</h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.3rem; color: #B3B3B3; margin-bottom: 3rem;'>"
        "When Simple Baselines Beat Machine Learning"
        "</p>",
        unsafe_allow_html=True
    )
    
    # Key finding card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%); 
                    padding: 3rem 2rem; 
                    border-radius: 12px; 
                    text-align: center; 
                    color: white;
                    box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3);
                    margin-bottom: 2rem;'>
            <p style='font-size: 1rem; font-weight: 600; letter-spacing: 2px; 
                      text-transform: uppercase; margin: 0; opacity: 0.9;'>
                Key Finding
            </p>
            <h1 style='font-size: 5rem; font-weight: 700; margin: 1rem 0; color: white;'>28%</h1>
            <p style='font-size: 1.2rem; margin: 0; opacity: 0.95;'>
                Simple baseline outperforms best AI model
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics row - Equal width columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Control HitRate",
            value="0.420",
            delta="WINNER",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Best AI HitRate",
            value="0.312",
            delta="-26%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Models Tested",
            value="5",
            help="Control, SVD Pure, SVD Hybrid, KNN Pure, KNN Hybrid"
        )
    
    with col4:
        st.metric(
            label="Conditions Tested",
            value="13",
            help="Temporal patterns, User segments, Movie types"
        )
    
    st.markdown("---")
    
    # Main comparison chart with filter
    st.markdown('<p class="section-header">Model Performance Comparison</p>', 
                unsafe_allow_html=True)
    
    # Add interactive filter
    col1, col2 = st.columns([3, 1])
    with col2:
        show_models = st.multiselect(
            "Select models to display:",
            options=results['overall']['Model'].tolist(),
            default=results['overall']['Model'].tolist(),
            key="exec_summary_filter"
        )
    
    if show_models:
        df_filtered = results['overall'][results['overall']['Model'].isin(show_models)]
        
        # Color mapping: Control green, AI models red/orange
        model_colors = {
            'Control': COLORS['success'],
            'SVD Pure': COLORS['error'],
            'SVD Hybrid': COLORS['warning'],
            'KNN Pure': COLORS['error'],
            'KNN Hybrid': COLORS['warning']
        }
        colors = [model_colors[model] for model in df_filtered['Model']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_filtered['Model'],
            y=df_filtered['HitRate@10'],
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            text=df_filtered['HitRate@10'].round(3),
            textposition='outside',
            textfont=dict(size=14, color='white'),
            name='HitRate@10',
            hovertemplate='<b>%{x}</b><br>HitRate: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Engagement Performance: Control Dominates",
                'font': {'size': 20, 'color': 'white', 'family': 'Arial, sans-serif'}
            },
            yaxis_title="HitRate@10 (Higher is Better)",
            height=500,
            showlegend=False,
            template="plotly_dark",
            font=dict(size=12, color='white'),
            plot_bgcolor=COLORS['bg_dark'],
            paper_bgcolor=COLORS['bg_dark'],
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                color='white'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                color='white'
            ),
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        fig.add_hline(
            y=0.420, 
            line_dash="dash", 
            line_color=COLORS['success'], 
            line_width=2,
            annotation_text="Control Baseline",
            annotation_position="right",
            annotation=dict(font=dict(color='white', size=12))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        try:
            create_download_button(fig, "model_performance_comparison.png")
        except:
            st.caption("Note: Download requires kaleido package. Install with: pip install kaleido")
    
    else:
        st.warning("Please select at least one model to display")
    
    # Key insights section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
        
        with st.expander("See detailed insights", expanded=True):
            st.markdown("""
            **What We Discovered:**
            
            - **Genre preferences dominate** user engagement more than collaborative patterns
            - AI models optimize rating prediction accuracy, not engagement (metric mismatch)
            - Feature engineering improves AI by 15-16%, but still insufficient
            - Control wins across **all** tested conditions:
              - Temporal: Weekdays, weekends, all seasons
              - Users: Light, casual, active, heavy raters
              - Movies: New releases, classics, popular titles
            """)
    
    with col2:
        st.markdown('<p class="section-header">Business Impact</p>', unsafe_allow_html=True)
        
        impact_data = pd.DataFrame({
            'Metric': ['Engagement', 'Inference Time', 'Training Cost', 'Interpretability'],
            'Control': ['0.420', '<1ms', '$0', 'Clear'],
            'Best AI': ['0.312', '28s-21min', 'Compute + time', 'Black box'],
            'Advantage': ['+28%', '1000x faster', 'Free', 'Debuggable']
        })
        
        st.dataframe(
            impact_data,
            use_container_width=True,
            hide_index=True
        )
        
        st.success("**Recommendation:** Deploy Control baseline immediately.")
    
    # Key findings - copy to clipboard
    st.markdown("---")
    st.markdown('<p class="section-header">Key Findings Summary</p>', unsafe_allow_html=True)
    
    key_findings = """
    AI Personalization Analysis - Key Findings:
    
    1. Control baseline achieves 28% better engagement than best AI model (0.420 vs 0.312 HitRate)
    2. Genre preferences dominate user engagement more than collaborative filtering patterns
    3. AI optimizes rating accuracy (RMSE) but business needs engagement (HitRate) - metric mismatch
    4. Feature engineering improves AI by 15-16% but cannot close the 26-28% gap
    5. Control wins across ALL 13 tested conditions (temporal, user segments, movie types)
    6. Computational cost: AI requires 1000x more resources for worse results
    7. Recommendation: Deploy simple genre-based baseline over complex AI models
    """
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.text_area("Copy these findings:", key_findings, height=200)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Copy the findings from the text box for presentations or reports")
    
    # Navigation hint
    st.markdown("---")
    st.info("Use the sidebar navigation to explore detailed results from each experiment")
    # ============================================================
# PAGE 2: EXPERIMENT 1 - BASELINE COMPARISON
# ============================================================

elif page == "Experiment 1: Baselines":
    
    st.markdown('<h1 class="main-header">Experiment 1: Baseline Model Comparison</h1>', 
                unsafe_allow_html=True)
    
    # Research question
    st.markdown(f"""
    <div style='background-color: {COLORS['bg_card']}; 
                padding: 1.5rem; 
                border-radius: 8px; 
                border-left: 4px solid {COLORS['primary']};
                margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; margin: 0; color: {COLORS['text']};'>
            <strong>Research Question:</strong> Can optimized AI models (SVD, KNN) beat a simple genre-based baseline?
        </p>
        <p style='font-size: 1.1rem; margin: 0.5rem 0 0 0; color: {COLORS['error']};'>
            <strong>Answer:</strong> No. Control wins by 37-38%.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Methodology expander
    with st.expander("Methodology Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Dataset:**
            - MovieLens 1M
            - 1,000,209 ratings from 6,040 users on 3,883 movies
            - Temporal split: 60% train / 20% validation / 20% test
            - Ensures no data leakage from future into past
            """)
        
        with col2:
            st.markdown("""
            **Models:**
            - **Control:** Popular movies within user's preferred genres
            - **SVD:** Matrix factorization (24 configurations tested)
            - **KNN:** Collaborative filtering (6 configurations tested)
            - All models properly hyperparameter tuned
            """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overall performance table
    st.markdown('<p class="section-header">Overall Performance</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        metric_choice = st.radio(
            "Select metric:",
            ["HitRate@10", "MRR@10", "Training Time (s)", "Inference Time (s)"],
            key="exp1_metric"
        )
    
    df_display = results['overall'].copy()
    df_display['HitRate@10'] = df_display['HitRate@10'].round(3)
    df_display['MRR@10'] = df_display['MRR@10'].round(3)
    
    # Style the dataframe - highlight best performance
    if metric_choice in ['HitRate@10', 'MRR@10']:
        styled_df = df_display.style.highlight_max(subset=[metric_choice], color='rgba(70, 211, 105, 0.3)')
    else:
        styled_df = df_display.style.highlight_min(subset=[metric_choice], color='rgba(70, 211, 105, 0.3)')
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=250
    )
    
    st.markdown("---")
    
    # Segment analysis
    st.markdown('<p class="section-header">Performance by User Segment</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **K-Means Clustering** identified 4 distinct user segments based on:
    - Number of ratings
    - Average rating score
    - Genre diversity
    - Rating patterns over time
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive toggle for segment comparison
    show_gap_analysis = st.checkbox("Show performance gap analysis", value=False, key="exp1_gap")
    
    seg_df = results['segments'].copy()
    seg_df['Gap (%)'] = ((seg_df['AI Best'] - seg_df['Control']) / seg_df['Control'] * 100).round(1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Control',
        x=seg_df['Segment'],
        y=seg_df['Control'],
        marker=dict(color=COLORS['success'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=seg_df['Control'].round(3),
        textposition='outside',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>Control</b><br>Segment: %{x}<br>HitRate: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Best AI Hybrid',
        x=seg_df['Segment'],
        y=seg_df['AI Best'],
        marker=dict(color=COLORS['warning'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=seg_df['AI Best'].round(3),
        textposition='outside',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>Best AI</b><br>Segment: %{x}<br>HitRate: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        yaxis_title="HitRate@10",
        template="plotly_dark",
        plot_bgcolor=COLORS['bg_dark'],
        paper_bgcolor=COLORS['bg_dark'],
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        ),
        margin=dict(t=60, b=60, l=60, r=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    try:
        create_download_button(fig, "segment_performance_exp1.png")
    except:
        pass
    
    # Show gap analysis if toggled
    if show_gap_analysis:
        st.markdown("### Performance Gap by Segment")
        gap_fig = go.Figure()
        
        gap_fig.add_trace(go.Bar(
            x=seg_df['Segment'],
            y=seg_df['Gap (%)'],
            marker=dict(color=COLORS['error']),
            text=seg_df['Gap (%)'].astype(str) + '%',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Gap: %{y:.1f}%<extra></extra>'
        ))
        
        gap_fig.update_layout(
            title="AI Performance Gap vs Control (Negative = AI loses)",
            yaxis_title="Gap (%)",
            height=400,
            template="plotly_dark",
            plot_bgcolor=COLORS['bg_dark'],
            paper_bgcolor=COLORS['bg_dark'],
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
        )
        
        gap_fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
        
        st.plotly_chart(gap_fig, use_container_width=True)
    
    # Segment insights
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Control wins across ALL segments:**
        - Heavy raters (351 ratings): **75% HitRate** (best performance)
        - Active users (105 ratings): **52% HitRate**
        - Even power users with rich history prefer genre-based recommendations
        - Consistent performance regardless of user sophistication
        """)
    
    with col2:
        st.error("""
        **AI consistently underperforms:**
        - Heavy raters: -25% vs Control
        - Active users: -30% vs Control
        - Casual users: -17% vs Control
        - Genre preferences dominate regardless of user type or activity level
        """)

# ============================================================
# PAGE 3: EXPERIMENT 2 - FEATURE ENGINEERING
# ============================================================

elif page == "Experiment 2: Features":
    
    st.markdown('<h1 class="main-header">Experiment 2: Feature Engineering</h1>', 
                unsafe_allow_html=True)
    
    # Research question
    st.markdown(f"""
    <div style='background-color: {COLORS['bg_card']}; 
                padding: 1.5rem; 
                border-radius: 8px; 
                border-left: 4px solid {COLORS['primary']};
                margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; margin: 0; color: {COLORS['text']};'>
            <strong>Research Question:</strong> Can rich features help AI models compete with the baseline?
        </p>
        <p style='font-size: 1.1rem; margin: 0.5rem 0 0 0; color: {COLORS['warning']};'>
            <strong>Answer:</strong> Features help (+15-16%) but AI still loses by 26-28%.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features engineered
    st.markdown('<p class="section-header">Features Engineered</p>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown(f"<h4 style='color: {COLORS['primary']};'>Temporal Features</h4>", 
                       unsafe_allow_html=True)
            st.markdown("""
            - Movie release year
            - Rating season (Winter/Spring/Summer/Fall)
            - Day of week (weekday vs weekend)
            - Movie age at rating (recency preference)
            """)
    
    with col2:
        with st.container():
            st.markdown(f"<h4 style='color: {COLORS['primary']};'>User Behavioral</h4>", 
                       unsafe_allow_html=True)
            st.markdown("""
            - Rating velocity (ratings per month)
            - Genre diversity (distinct genres rated)
            - New movie preference (% recent films)
            - Weekend rater tendency (Sat/Sun activity)
            """)
    
    with col3:
        with st.container():
            st.markdown(f"<h4 style='color: {COLORS['primary']};'>Interaction Features</h4>", 
                       unsafe_allow_html=True)
            st.markdown("""
            - Genre match scores (user-movie overlap)
            - 23M+ precomputed scores via matrix multiplication
            - Vectorized numpy operations for efficiency
            - O(1) lookup during inference
            """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature impact chart
    st.markdown('<p class="section-header">Impact of Feature Engineering</p>', 
                unsafe_allow_html=True)
    
    # Toggle for different visualizations
    viz_type = st.radio(
        "Visualization type:",
        ["Grouped Bars", "Improvement Arrows"],
        horizontal=True,
        key="exp2_viz"
    )
    
    feat_df = results['feature_impact']
    
    if viz_type == "Grouped Bars":
        fig = go.Figure()
        
        # Pure models (red)
        fig.add_trace(go.Bar(
            name='Pure (No Features)',
            x=feat_df['Model'],
            y=feat_df['Pure'],
            marker=dict(color=COLORS['error'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
            text=feat_df['Pure'].round(3),
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>Pure Model</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
        ))
        
        # Hybrid models (orange)
        fig.add_trace(go.Bar(
            name='Hybrid (+Features)',
            x=feat_df['Model'],
            y=feat_df['Hybrid'],
            marker=dict(color=COLORS['warning'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
            text=feat_df['Hybrid'].round(3),
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>Hybrid Model</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
        ))
        
        # Control baseline (green)
        fig.add_trace(go.Bar(
            name='Control Baseline',
            x=feat_df['Model'],
            y=[0.420, 0.420],
            marker=dict(color=COLORS['success'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
            text=[0.420, 0.420],
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>Control</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            yaxis_title="HitRate@10",
            template="plotly_dark",
            plot_bgcolor=COLORS['bg_dark'],
            paper_bgcolor=COLORS['bg_dark'],
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            ),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
    else:  # Improvement Arrows
        fig = go.Figure()
        
        for idx, row in feat_df.iterrows():
            # Line showing improvement
            fig.add_trace(go.Scatter(
                x=[row['Model'], row['Model']],
                y=[row['Pure'], row['Hybrid']],
                mode='lines+markers',
                line=dict(color=COLORS['success'], width=3),
                marker=dict(size=10, color=[COLORS['error'], COLORS['warning']]),
                name=row['Model'],
                showlegend=True,
                hovertemplate=f'<b>{row["Model"]}</b><br>Pure: {row["Pure"]:.3f}<br>Hybrid: {row["Hybrid"]:.3f}<br>Improvement: {row["Improvement"]}<extra></extra>'
            ))
        
        # Add control line
        fig.add_hline(y=0.420, line_dash="dash", line_color=COLORS['success'], line_width=2,
                     annotation_text="Control Baseline", annotation_position="right")
        
        fig.update_layout(
            height=500,
            yaxis_title="HitRate@10",
            title="Feature Engineering Impact: Before → After",
            template="plotly_dark",
            plot_bgcolor=COLORS['bg_dark'],
            paper_bgcolor=COLORS['bg_dark'],
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            margin=dict(t=60, b=60, l=60, r=40)
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    try:
        create_download_button(fig, "feature_impact_exp2.png")
    except:
        pass
    
    # Results table
    st.markdown("---")
    st.markdown('<p class="section-header">Feature Engineering Results</p>', 
                unsafe_allow_html=True)
    
    results_table = pd.DataFrame({
        'Model': ['Control', 'SVD Pure', 'SVD Hybrid', 'KNN Pure', 'KNN Hybrid'],
        'HitRate@10': [0.420, 0.263, 0.302, 0.269, 0.312],
        'Improvement': ['Baseline', '—', '+15%', '—', '+16%'],
        'vs Control': ['Winner', '-37%', '-28%', '-36%', '-26%']
    })
    
    st.dataframe(results_table, use_container_width=True, hide_index=True, height=250)
    
    # Key insights
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Features DO Help AI:**
        - SVD: +15% improvement (0.263 → 0.302)
        - KNN: +16% improvement (0.269 → 0.312)
        - Genre matching is the key signal captured
        - Temporal and behavioral features add value
        - Best performance at α=1.0 (heavy genre weighting)
        """)
    
    with col2:
        st.error("""
        **But NOT Enough to Beat Baseline:**
        - Best hybrid still loses by 26% to Control
        - Control already uses genres optimally (direct matching)
        - Adding genre features makes AI converge toward Control's approach
        - Computational cost (1000x slower) not justified by performance
        - Operational complexity adds no business value
        """)
        # ============================================================
# PAGE 4: CONDITIONAL ANALYSIS
# ============================================================

elif page == "Conditional Analysis":
    
    st.markdown('<h1 class="main-header">Conditional Analysis: When Does AI Win?</h1>', 
                unsafe_allow_html=True)
    
    # Research question
    st.markdown(f"""
    <div style='background-color: {COLORS['bg_card']}; 
                padding: 1.5rem; 
                border-radius: 8px; 
                border-left: 4px solid {COLORS['primary']};
                margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; margin: 0; color: {COLORS['text']};'>
            <strong>Research Question:</strong> Are there specific conditions where AI outperforms Control?
        </p>
        <p style='font-size: 1.1rem; margin: 0.5rem 0 0 0; color: {COLORS['error']};'>
            <strong>Answer:</strong> No. Control wins across ALL tested conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Select Analysis Dimension:",
        ["Temporal Patterns", "User Segments", "Movie Characteristics"],
        label_visibility="visible"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================================
    # TEMPORAL ANALYSIS
    # ============================================================
    
    if analysis_type == "Temporal Patterns":
        
        st.markdown('<p class="section-header">Performance by Temporal Context</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        **Hypothesis:** AI might win during specific times (weekdays vs weekends, different seasons)
        
        **Result:** Control wins across ALL temporal contexts (6/6 conditions tested)
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive filter for temporal contexts
        temp_df = results['temporal']
        
        col1, col2 = st.columns([3, 1])
        with col2:
            show_contexts = st.multiselect(
                "Select contexts:",
                options=temp_df['Context'].tolist(),
                default=temp_df['Context'].tolist(),
                key="temporal_filter"
            )
        
        if show_contexts:
            temp_filtered = temp_df[temp_df['Context'].isin(show_contexts)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Control',
                x=temp_filtered['Context'],
                y=temp_filtered['Control'],
                marker=dict(color=COLORS['success'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
                text=temp_filtered['Control'].round(3),
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>Control</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='Best AI Hybrid',
                x=temp_filtered['Context'],
                y=temp_filtered['AI Best'],
                marker=dict(color=COLORS['warning'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
                text=temp_filtered['AI Best'].round(3),
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>Best AI</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                barmode='group',
                height=500,
                yaxis_title="HitRate@10",
                template="plotly_dark",
                plot_bgcolor=COLORS['bg_dark'],
                paper_bgcolor=COLORS['bg_dark'],
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                ),
                margin=dict(t=60, b=60, l=60, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            try:
                create_download_button(fig, "temporal_analysis.png")
            except:
                pass
            
            # Statistics table
            st.markdown("### Temporal Performance Statistics")
            temp_stats = temp_filtered.copy()
            temp_stats['Gap (%)'] = ((temp_stats['AI Best'] - temp_stats['Control']) / temp_stats['Control'] * 100).round(1)
            st.dataframe(temp_stats, use_container_width=True, hide_index=True)
        
        st.info("""
        **Finding:** Time of day/year doesn't change genre preferences
        
        - Worst AI gap: Winter (-28.3% vs Control)
        - Best AI performance: Spring (still -26% vs Control)
        - Genre-based recommendations work consistently across temporal patterns
        - No seasonal or weekday/weekend advantage for AI models
        """)
    
    # ============================================================
    # USER SEGMENT ANALYSIS
    # ============================================================
    
    elif analysis_type == "User Segments":
        
        st.markdown('<p class="section-header">Performance by User Type</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        **Hypothesis:** AI might excel for power users with rich interaction history
        
        **Result:** Control wins for ALL user types, including heavy raters (4/4 segments tested)
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        seg_df = results['segments'].copy()
        seg_df['Gap (%)'] = ((seg_df['AI Best'] - seg_df['Control']) / seg_df['Control'] * 100).round(1)
        
        # Visualization toggle
        seg_viz = st.radio(
            "Select visualization:",
            ["Performance Comparison", "Gap Analysis", "Both"],
            horizontal=True,
            key="seg_viz"
        )
        
        if seg_viz in ["Performance Comparison", "Both"]:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Control',
                x=seg_df['Segment'],
                y=seg_df['Control'],
                marker=dict(color=COLORS['success'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
                text=seg_df['Control'].round(3),
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>Control</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='Best AI',
                x=seg_df['Segment'],
                y=seg_df['AI Best'],
                marker=dict(color=COLORS['warning'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
                text=seg_df['AI Best'].round(3),
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>Best AI</b><br>%{x}<br>HitRate: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                barmode='group',
                height=500,
                yaxis_title="HitRate@10",
                template="plotly_dark",
                plot_bgcolor=COLORS['bg_dark'],
                paper_bgcolor=COLORS['bg_dark'],
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                ),
                margin=dict(t=60, b=60, l=60, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            try:
                create_download_button(fig, "segment_analysis.png")
            except:
                pass
        
        if seg_viz in ["Gap Analysis", "Both"]:
            st.markdown("<br>", unsafe_allow_html=True)
            
            gap_fig = go.Figure()
            
            gap_fig.add_trace(go.Bar(
                x=seg_df['Segment'],
                y=seg_df['Gap (%)'],
                marker=dict(
                    color=COLORS['error'],
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                text=seg_df['Gap (%)'].astype(str) + '%',
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>%{x}</b><br>Gap: %{y:.1f}%<extra></extra>'
            ))
            
            gap_fig.update_layout(
                title="AI Performance Gap vs Control (All Negative = AI Loses)",
                yaxis_title="Gap (%)",
                height=400,
                template="plotly_dark",
                plot_bgcolor=COLORS['bg_dark'],
                paper_bgcolor=COLORS['bg_dark'],
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                margin=dict(t=60, b=60, l=60, r=40)
            )
            
            gap_fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
            
            st.plotly_chart(gap_fig, use_container_width=True)
        
        # Segment details - expandable
        st.markdown('<p class="section-header">Detailed Segment Analysis</p>', 
                    unsafe_allow_html=True)
        
        for _, row in seg_df.iterrows():
            with st.expander(f"Analysis: {row['Segment']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Control Performance", f"{row['Control']:.3f}")
                with col2:
                    st.metric("Best AI Performance", f"{row['AI Best']:.3f}")
                with col3:
                    st.metric("Performance Gap", f"{row['Gap (%)']}%", delta=f"{row['Gap (%)']}%", delta_color="inverse")
        
        st.error("""
        **Finding:** Even power users (351 avg ratings) prefer genre-based recommendations
        
        - Heavy raters: Control 0.750 vs AI 0.562 (-25%)
        - Active users: Control 0.521 vs AI 0.367 (-30%)
        - Genre preferences dominate regardless of user sophistication or activity level
        - No user segment benefits from AI personalization over simple baseline
        """)
    
    # ============================================================
    # MOVIE CHARACTERISTICS ANALYSIS
    # ============================================================
    
    else:  # Movie Characteristics
        
        st.markdown('<p class="section-header">Performance by Movie Type</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        **Hypothesis:** AI might win for niche or new releases
        
        **Result:** Control wins for popular movies (where it matters). AI only wins for niche content with ~0% hit rates.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        movie_df = pd.DataFrame({
            'Category': ['New & Popular', 'Classic & Popular', 'New & Niche', 'Classic & Niche'],
            'Control': [0.066, 0.055, 0.000, 0.000],
            'AI Best': [0.023, 0.039, 0.0001, 0.005],
            'Hit Rate': ['6.6%', '5.5%', '~0%', '~0%']
        })
        
        # Add interactive selection
        show_niche = st.checkbox("Include niche movies (very low engagement)", value=True, key="show_niche")
        
        if not show_niche:
            movie_df = movie_df[movie_df['Category'].str.contains('Popular')]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Control',
            x=movie_df['Category'],
            y=movie_df['Control'],
            marker=dict(color=COLORS['success'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
            text=movie_df['Hit Rate'],
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>Control</b><br>%{x}<br>Hit Rate: %{text}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='AI Best',
            x=movie_df['Category'],
            y=movie_df['AI Best'],
            marker=dict(color=COLORS['warning'], line=dict(color='rgba(255,255,255,0.2)', width=1)),
            hovertemplate='<b>AI Best</b><br>%{x}<br>Hit Rate: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            yaxis_title="Hit Rate",
            template="plotly_dark",
            plot_bgcolor=COLORS['bg_dark'],
            paper_bgcolor=COLORS['bg_dark'],
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            ),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        try:
            create_download_button(fig, "movie_characteristics_analysis.png")
        except:
            pass
        
        st.warning("""
        **Finding:** AI "wins" only for niche movies with negligible impact
        
        - Control design limits recommendations to popular movies (by design)
        - AI slightly better for long-tail content (0.5% vs 0%)
        - But niche movie recommendations have ~0% engagement anyway
        - Popular movies drive vast majority of actual user engagement
        - This is not a meaningful advantage for AI
        """)
    
    # Conditional analysis summary
    st.markdown("---")
    st.markdown('<p class="section-header">Conditional Analysis Summary</p>', 
                unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        'Dimension': ['Temporal Patterns', 'User Segments', 'Movie Types (Popular)'],
        'Conditions Tested': ['6 (weekday/weekend, 4 seasons)', '4 (casual, active, heavy, minimal)', '4 (new/old, popular/niche)'],
        'AI Wins': ['0 / 6', '0 / 4', '0 / 2 (popular only)'],
        'Average Gap': ['-25%', '-24%', '-48%']
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Conditions Tested", "13")
    with col2:
        st.metric("AI Wins", "0", delta="Control dominates", delta_color="inverse")
    with col3:
        st.metric("Average Gap", "-32%", help="Average performance gap across all conditions")
    
    st.error("""
    **Conclusion:** No conditional advantage found for AI models
    
    - Tested 13 distinct conditions across 3 dimensions
    - Control wins in every meaningful scenario
    - Only "wins" for AI are edge cases with ~0% hit rates (not useful)
    - Genre-based recommendations are universally effective
    - No niche identified where AI complexity is justified by performance
    """)
    # ============================================================
# PAGE 5: BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":
    
    st.markdown('<h1 class="main-header">Business Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # Main recommendation card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {COLORS['success']} 0%, #2ea85c 100%); 
                padding: 3rem 2rem; 
                border-radius: 12px; 
                text-align: center; 
                color: white;
                box-shadow: 0 8px 32px rgba(70, 211, 105, 0.3);
                margin-bottom: 3rem;'>
        <p style='font-size: 1rem; font-weight: 600; letter-spacing: 2px; 
                  text-transform: uppercase; margin: 0; opacity: 0.9;'>
            Recommended Action
        </p>
        <h1 style='font-size: 3rem; font-weight: 700; margin: 1rem 0; color: white;'>
            Deploy the Control Baseline
        </h1>
        <p style='font-size: 1.2rem; margin: 0; opacity: 0.95;'>
            Simple genre-based popularity recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rationale section
    st.markdown('<p class="section-header">Rationale</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Why Control Wins")
        
        st.markdown("**Performance:**")
        st.markdown("""
        - 28% better engagement than best AI model
        - Consistent across all user segments
        - Robust across temporal patterns
        - Wins in every meaningful condition tested
        """)
        
        st.markdown("**Operational Efficiency:**")
        st.markdown("""
        - Sub-millisecond inference time
        - Zero training cost
        - No model maintenance required
        - Scales linearly with user base
        """)
        
        st.markdown("**Product Quality:**")
        st.markdown("""
        - Interpretable recommendations
        - Easy to debug and explain
        - No algorithmic bias concerns
        - Transparent to stakeholders
        """)
    
    with col2:
        st.markdown("### Why AI Falls Short")
        
        st.markdown("**Metric Mismatch:**")
        st.markdown("""
        - Optimizes rating prediction accuracy (RMSE)
        - Business needs engagement (clicks, watches)
        - Strong RMSE (0.88) leads to weak HitRate (0.26)
        - Training objective misaligned with business goal
        """)
        
        st.markdown("**Signal Already Captured:**")
        st.markdown("""
        - Genre preferences are the dominant signal
        - Control uses this signal optimally (direct matching)
        - AI learns same pattern but with overhead
        - Collaborative filtering adds no new information
        """)
        
        st.markdown("**Cost Not Justified:**")
        st.markdown("""
        - 1000x slower inference (28s-21min vs <1ms)
        - Requires ongoing model retraining
        - Operational complexity (pipelines, monitoring)
        - Delivers 26-28% worse user experience
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ROI Calculator (interactive)
    st.markdown("---")
    st.markdown('<p class="section-header">ROI Calculator</p>', unsafe_allow_html=True)
    
    with st.expander("Calculate deployment costs (interactive)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Control Baseline:**")
            users = st.number_input("Number of users:", value=10000, step=1000, key="roi_users")
            st.metric("Inference Cost", "$0")
            st.metric("Training Cost", "$0")
            st.metric("Maintenance Cost/Month", "$0")
            st.metric("Expected HitRate", "42%")
        
        with col2:
            st.markdown("**AI Model (SVD):**")
            compute_cost = st.number_input("Compute cost per hour ($):", value=2.0, step=0.5, key="compute_cost")
            
            training_hours = 45 / 3600  # 45 seconds to hours
            inference_hours = (users * 28) / 3600  # 28s per user to hours
            monthly_retraining = 4  # Retrain weekly
            
            training_cost = training_hours * compute_cost * monthly_retraining
            inference_cost = inference_hours * compute_cost * 30  # Daily inference
            
            st.metric("Inference Cost/Month", f"${inference_cost:.2f}")
            st.metric("Training Cost/Month", f"${training_cost:.2f}")
            st.metric("Maintenance Cost/Month", "$1000+")
            st.metric("Expected HitRate", "26%", delta="-38% vs Control", delta_color="inverse")
        
        st.error(f"**Cost difference:** AI costs ${inference_cost + training_cost + 1000:.2f}/month more while delivering worse results")
    
    # When to revisit AI
    st.markdown("---")
    st.markdown('<p class="section-header">When to Revisit AI</p>', 
                unsafe_allow_html=True)
    
    st.markdown("Consider AI models if/when these conditions change:")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **New Data Sources Available:**
        - Implicit signals (clicks, watch time, completion rates)
        - User demographic or psychographic data
        - Social/collaborative features (friend networks, shared lists)
        - Rich content metadata (plot summaries, cast, directors)
        """)
        
        st.info("""
        **Business Goals Shift:**
        - Diversity/serendipity prioritized over pure engagement
        - Cold-start problem becomes critical (many new users/items)
        - Long-tail content promotion becomes strategic priority
        - Exploration vs exploitation tradeoff changes
        """)
    
    with col2:
        st.info("""
        **Technical Landscape Changes:**
        - Computational cost becomes negligible (better hardware)
        - Real-time training becomes feasible
        - Hybrid approaches can be A/B tested efficiently
        - New algorithms significantly improve performance
        """)
        
        st.info("""
        **Market Dynamics Evolve:**
        - User behavior shifts significantly over time
        - Genre preferences become less predictive
        - Competitive pressure requires more sophistication
        - Regulatory requirements change recommendation logic
        """)
    
    # Implementation roadmap
    st.markdown("---")
    st.markdown('<p class="section-header">Implementation Roadmap</p>', 
                unsafe_allow_html=True)
    
    roadmap = pd.DataFrame({
        'Phase': ['Phase 1: Deploy Control', 'Phase 2: Monitor', 'Phase 3: Iterate', 'Phase 4: Experiment'],
        'Timeline': ['Week 1', 'Weeks 2-4', 'Month 2', 'Month 3+'],
        'Activities': [
            'Implement genre-based recommender, A/B test vs current system',
            'Track engagement metrics, Collect implicit signals (clicks, watch time)',
            'Refine genre matching logic, Add simple personalization (e.g., recency)',
            'Test AI models with new signals if baseline plateaus'
        ],
        'Success Criteria': [
            '>20% engagement lift',
            'Metrics stable or improving',
            'Incremental improvements visible',
            'AI beats Control by >10%'
        ]
    })
    
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
    
    # Download roadmap as CSV
    csv = roadmap.to_csv(index=False)
    st.download_button(
        label="Download Roadmap as CSV",
        data=csv,
        file_name="implementation_roadmap.csv",
        mime="text/csv"
    )
    
    # Key lessons
    st.markdown("---")
    st.markdown('<p class="section-header">Key Lessons for ML Practitioners</p>', 
                unsafe_allow_html=True)
    
    # Interactive lesson selector
    lesson_category = st.radio(
        "Select category:",
        ["When Simple Baselines Win", "Best Practices Demonstrated", "Both"],
        horizontal=True,
        key="lesson_category"
    )
    
    if lesson_category in ["When Simple Baselines Win", "Both"]:
        st.success("""
        **When Simple Baselines Win:**
        
        1. **Domain signal is clear and explicit** — Genre preferences are directly observable in user history; no need for latent factor discovery
        
        2. **Metric mismatch between training and deployment** — Optimizing RMSE doesn't improve engagement; always align model objectives with business metrics
        
        3. **Cost-benefit analysis favors simplicity** — 1000x computational overhead + 28% worse performance = bad ROI; operational complexity not justified
        """)
    
    if lesson_category in ["Best Practices Demonstrated", "Both"]:
        st.warning("""
        **Best Practices Demonstrated:**
        
        - Start with strong baselines before deploying complex models
        - Use business-relevant metrics (engagement) over technical metrics (accuracy)
        - Conduct rigorous conditional analysis to find when each approach wins
        - Track computational cost vs marginal performance gains
        - Be willing to recommend against AI when evidence supports it
        - Proper temporal splits prevent data leakage and overfitting
        - Hyperparameter tuning ensures fair model comparison
        - Test across multiple user segments and contexts
        - Document decision-making process for transparency
        - Consider operational complexity in production environments
        """)
    
    # Executive summary for stakeholders
    st.markdown("---")
    st.markdown('<p class="section-header">Executive Summary for Stakeholders</p>', 
                unsafe_allow_html=True)
    
    executive_summary = """
    EXECUTIVE SUMMARY: AI PERSONALIZATION ANALYSIS
    
    RECOMMENDATION: Deploy simple genre-based baseline over AI models
    
    KEY FINDINGS:
    • Control baseline achieves 42% engagement vs 31% for best AI model (+28% advantage)
    • Tested 5 models across 13 different conditions - Control wins everywhere
    • AI models cost 1000x more to run with worse user experience
    • Genre preferences dominate engagement - simple matching beats complex ML
    
    BUSINESS IMPACT:
    • Higher user engagement (28% improvement)
    • Zero infrastructure cost
    • Instant recommendations (<1ms)
    • Easy to explain and debug
    • No training or maintenance overhead
    
    WHEN TO REVISIT AI:
    • New data sources available (clicks, watch time, demographics)
    • Business goals shift (diversity over engagement)
    • Computational costs become negligible
    • Genre preferences become less predictive
    
    NEXT STEPS:
    1. Deploy genre-based recommender (Week 1)
    2. A/B test and monitor engagement (Weeks 2-4)
    3. Iterate on genre matching logic (Month 2)
    4. Re-evaluate AI if baseline plateaus (Month 3+)
    
    CONFIDENCE LEVEL: High
    • Rigorous methodology (proper train/val/test splits)
    • Extensive testing (13 conditions, 4 user segments)
    • Clear performance gap (26-28% across all conditions)
    • Aligned with published research on strong baselines
    """
    
    st.text_area("Copy for stakeholder presentation:", executive_summary, height=400)
    
    # Download as text file
    st.download_button(
        label="Download Executive Summary",
        data=executive_summary,
        file_name="executive_summary.txt",
        mime="text/plain"
    )
    
    # Final CTA
    st.markdown("---")
    st.markdown(f"""
    <div style='background-color: {COLORS['bg_card']}; 
                padding: 2rem; 
                border-radius: 8px; 
                text-align: center;
                border: 2px solid {COLORS['primary']};'>
        <h3 style='color: {COLORS['text']}; margin-top: 0;'>Ready to Deploy?</h3>
        <p style='color: {COLORS['text_secondary']}; font-size: 1.1rem;'>
            Use the sidebar to review detailed analysis or download key findings for your team.
        </p>
        <p style='color: {COLORS['text_secondary']}; margin-bottom: 0;'>
            Questions? Contact: <strong style='color: {COLORS['primary']};'>rahul.yatham@example.com</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER (Appears on all pages)
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

# Footer with stats and credits
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='text-align: center;'>
        <p style='color: {COLORS['text_secondary']}; margin: 0;'><strong>Dataset</strong></p>
        <p style='color: {COLORS['text_secondary']}; margin: 0; font-size: 0.9rem;'>MovieLens 1M</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='text-align: center;'>
        <p style='color: {COLORS['text_secondary']}; margin: 0;'><strong>Models Tested</strong></p>
        <p style='color: {COLORS['text_secondary']}; margin: 0; font-size: 0.9rem;'>5 (Control, SVD, KNN + variants)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='text-align: center;'>
        <p style='color: {COLORS['text_secondary']}; margin: 0;'><strong>Author</strong></p>
        <p style='color: {COLORS['text_secondary']}; margin: 0; font-size: 0.9rem;'>Rahul Yatham</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    f"<p style='text-align: center; color: {COLORS['text_secondary']}; font-size: 0.9rem;'>"
    "Built with Streamlit • "
    "<a href='https://www.linkedin.com/in/rahul-yatham-15874a126/' style='color: " + COLORS['primary'] + ";'>LinkedIn</a> • "
    "<a href='https://github.com/yourusername' style='color: " + COLORS['primary'] + ";'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)
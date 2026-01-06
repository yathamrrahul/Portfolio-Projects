import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Personalization Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Executive Summary", 
     "📈 Experiment 1: Baselines", 
     "🔬 Experiment 2: Features",
     "🔍 Conditional Analysis",
     "💼 Business Recommendations"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**AI Personalization Analysis**\n\n"
    "Rigorous comparison of AI models vs. simple baselines "
    "for movie recommendations.\n\n"
    "📊 Dataset: MovieLens 1M\n"
    "📝 1M ratings, 6K users, 4K movies"
)

# Load data
@st.cache_data
def load_results():
    """Load pre-computed results from experiments"""
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

results = load_results()

# ============================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================
if page == "🏠 Executive Summary":
    st.markdown('<h1 class="main-header">🎬 AI Personalization Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown("### When Simple Baselines Beat Machine Learning")
    
    st.markdown("---")
    
    # Key Finding - Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
            <h2 style='color: white; margin: 0;'>🎯 Key Finding</h2>
            <h1 style='font-size: 4rem; margin: 1rem 0; color: white;'>28%</h1>
            <p style='font-size: 1.3rem; margin: 0;'>
                Simple baseline outperforms best AI model
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("##")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Control HitRate", "0.420", delta="Winner ✅", delta_color="normal")
    with col2:
        st.metric("Best AI HitRate", "0.312", delta="-26%", delta_color="inverse")
    with col3:
        st.metric("Models Tested", "5", help="Control, SVD Pure, SVD Hybrid, KNN Pure, KNN Hybrid")
    with col4:
        st.metric("Conditions Tested", "13", help="Temporal patterns, User segments, Movie types")
    
    st.markdown("---")
    
    # Main comparison chart
    st.subheader("📊 Model Performance Comparison")
    
    df = results['overall']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e74c3c', '#f39c12']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Model'],
        y=df['HitRate@10'],
        marker_color=colors,
        text=df['HitRate@10'].round(3),
        textposition='outside',
        textfont=dict(size=14, color='black'),
        name='HitRate@10'
    ))
    
    fig.update_layout(
        title="Engagement Performance: Control Dominates",
        yaxis_title="HitRate@10 (Higher is Better)",
        height=500,
        showlegend=False,
        template="plotly_white",
        font=dict(size=12)
    )
    
    fig.add_hline(y=0.420, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="Control Baseline", annotation_position="right")
    
    st.plotly_chart(fig, width='stretch')
    
    # Side-by-side: Insights & Impact
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Key Insights")
        st.markdown("""
        **✅ What We Discovered:**
        - **Genre preferences dominate** user engagement
        - AI optimizes wrong metric (rating accuracy ≠ engagement)
        - Feature engineering helps AI (+15-16%) but **not enough**
        - Control wins across **ALL** tested conditions:
          - ✓ Temporal: Weekdays, weekends, all seasons
          - ✓ Users: Light, casual, active, heavy raters
          - ✓ Movies: New releases, classics, popular titles
        """)
    
    with col2:
        st.subheader("💼 Business Impact")
        st.markdown("""
        **Why Control Baseline Wins:**
        
        | Metric | Control | Best AI | Advantage |
        |--------|---------|---------|-----------|
        | **Engagement** | 0.420 | 0.312 | +28% |
        | **Inference Time** | <1ms | 28s-21min | 1000x faster |
        | **Training Cost** | $0 | Compute + time | Free |
        | **Interpretability** | ✅ Clear | ❌ Black box | Debuggable |
        
        **Recommendation:** Deploy Control baseline immediately.
        """)
    
    # Quick navigation
    st.markdown("---")
    st.info("👈 **Use the sidebar** to explore detailed results from each experiment")

# ============================================================
# PAGE 2: EXPERIMENT 1
# ============================================================
elif page == "📈 Experiment 1: Baselines":
    st.title("📈 Experiment 1: Baseline Model Comparison")
    
    st.markdown("""
    **Research Question:** Can optimized AI models (SVD, KNN) beat a simple genre-based baseline?
    
    **Answer:** ❌ No. Control wins by 37-38%.
    """)
    
    st.markdown("---")
    
    # Methodology
    with st.expander("🔬 Methodology Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Dataset:**
            - MovieLens 1M
            - 1M ratings, 6K users, 4K movies
            - Temporal split: 60% train / 20% val / 20% test
            """)
        with col2:
            st.markdown("""
            **Models:**
            - Control: Popular movies in user's preferred genres
            - SVD: Matrix factorization (hyperparameter tuned)
            - KNN: Collaborative filtering (hyperparameter tuned)
            """)
    
    st.markdown("##")
    
    # Performance table
    st.subheader("📊 Overall Performance")
    
    df_display = results['overall'].copy()
    df_display['HitRate@10'] = df_display['HitRate@10'].round(3)
    df_display['MRR@10'] = df_display['MRR@10'].round(3)
    
    # Highlight winner
    st.dataframe(
        df_display.style.highlight_max(subset=['HitRate@10'], color='lightgreen'),
        width='stretch', 
        height=250
    )
    
    st.markdown("---")
    
    # Segment analysis
    st.subheader("👥 Performance by User Segment")
    
    st.markdown("""
    **K-Means Clustering** identified 4 distinct user segments based on:
    - Number of ratings
    - Average rating score
    - Genre diversity
    - Rating patterns
    """)
    
    seg_df = results['segments']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Control',
        x=seg_df['Segment'],
        y=seg_df['Control'],
        marker_color='#2ecc71',
        text=seg_df['Control'].round(3),
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig.add_trace(go.Bar(
        name='Best AI Hybrid',
        x=seg_df['Segment'],
        y=seg_df['AI Best'],
        marker_color='#f39c12',
        text=seg_df['AI Best'].round(3),
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        yaxis_title="HitRate@10",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Segment insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Control wins across ALL segments:**
        - Heavy raters (351 ratings): **75% HitRate** 🏆
        - Active users (105 ratings): **52% HitRate**
        - Even power users prefer genre-based recs
        """)
    
    with col2:
        st.error("""
        **AI consistently underperforms:**
        - Heavy raters: -25% vs Control
        - Active users: -30% vs Control
        - Genre preferences dominate regardless of user type
        """)

# ============================================================
# PAGE 3: EXPERIMENT 2
# ============================================================
elif page == "🔬 Experiment 2: Features":
    st.title("🔬 Experiment 2: Feature Engineering")
    
    st.markdown("""
    **Research Question:** Can rich features help AI models compete with the baseline?
    
    **Answer:** ✅ Features help (+15-16%) but ❌ AI still loses by 26-28%.
    """)
    
    st.markdown("---")
    
    # Features added
    st.subheader("🎯 Features Engineered")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⏰ Temporal Features**
        - Movie release year
        - Season (Winter/Spring/Summer/Fall)
        - Day of week
        - Movie age at rating
        """)
    
    with col2:
        st.markdown("""
        **👤 User Behavioral**
        - Rating velocity (ratings/month)
        - Genre diversity
        - New movie preference
        - Weekend rater tendency
        """)
    
    with col3:
        st.markdown("""
        **🔗 Interaction Features**
        - Genre match scores
        - 23M precomputed via matrix multiplication
        - Vectorized numpy operations
        """)
    
    st.markdown("---")
    
    # Feature impact
    st.subheader("📈 Impact of Feature Engineering")
    
    feat_df = results['feature_impact']
    
    fig = go.Figure()
    
    # Pure models
    fig.add_trace(go.Bar(
        name='Pure (No Features)',
        x=feat_df['Model'],
        y=feat_df['Pure'],
        marker_color='#e74c3c',
        text=feat_df['Pure'].round(3),
        textposition='outside'
    ))
    
    # Hybrid models
    fig.add_trace(go.Bar(
        name='Hybrid (+Features)',
        x=feat_df['Model'],
        y=feat_df['Hybrid'],
        marker_color='#f39c12',
        text=feat_df['Hybrid'].round(3),
        textposition='outside'
    ))
    
    # Control baseline
    fig.add_trace(go.Bar(
        name='Control Baseline',
        x=feat_df['Model'],
        y=[0.420, 0.420],
        marker_color='#2ecc71',
        text=[0.420, 0.420],
        textposition='outside'
    ))
    
    fig.update_layout(
        barmode='group',
        height=500,
        yaxis_title="HitRate@10",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Results table
    st.markdown("---")
    st.subheader("📊 Feature Engineering Results")
    
    results_table = pd.DataFrame({
        'Model': ['Control', 'SVD Pure', 'SVD Hybrid', 'KNN Pure', 'KNN Hybrid'],
        'HitRate@10': [0.420, 0.263, 0.302, 0.269, 0.312],
        'Improvement': ['Baseline', '-', '+15%', '-', '+16%'],
        'vs Control': ['Winner', '-37%', '-28%', '-36%', '-26%']
    })
    
    st.dataframe(results_table, width='stretch', height=250)
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Features DO Help AI:**
        - SVD: +15% improvement (0.263 → 0.302)
        - KNN: +16% improvement (0.269 → 0.312)
        - Genre matching is the key signal
        """)
    
    with col2:
        st.error("""
        **❌ But NOT Enough to Beat Baseline:**
        - Best hybrid still loses by 26%
        - Control already uses genres optimally
        - Computational cost not justified
        """)

# ============================================================
# PAGE 4: CONDITIONAL ANALYSIS
# ============================================================
elif page == "🔍 Conditional Analysis":
    st.title("🔍 Conditional Analysis: When Does AI Win?")
    
    st.markdown("""
    **Research Question:** Are there specific conditions where AI outperforms Control?
    
    **Answer:** ❌ No. Control wins across ALL tested conditions.
    """)
    
    st.markdown("---")
    
    # Analysis selector
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["⏰ Temporal Patterns", "👥 User Segments", "🎬 Movie Characteristics"]
    )
    
    st.markdown("##")
    
    if analysis_type == "⏰ Temporal Patterns":
        st.subheader("⏰ Performance by Temporal Context")
        
        st.markdown("""
        **Hypothesis:** AI might win during specific times (weekdays vs weekends, different seasons)
        
        **Result:** Control wins across ALL temporal contexts
        """)
        
        temp_df = results['temporal']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Control',
            x=temp_df['Context'],
            y=temp_df['Control'],
            marker_color='#2ecc71',
            text=temp_df['Control'].round(3),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Best AI Hybrid',
            x=temp_df['Context'],
            y=temp_df['AI Best'],
            marker_color='#f39c12',
            text=temp_df['AI Best'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            yaxis_title="HitRate@10",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.info("""
        **Finding:** Time of day/year doesn't change genre preferences
        - Worst AI gap: Winter (-28.3%)
        - Best AI performance: Spring (still -26%)
        - Genre-based recommendations work consistently
        """)
    
    elif analysis_type == "👥 User Segments":
        st.subheader("👥 Performance by User Type")
        
        st.markdown("""
        **Hypothesis:** AI might excel for power users with rich interaction history
        
        **Result:** Control wins for ALL user types, including heavy raters
        """)
        
        seg_df = results['segments']
        
        # Create gap calculation
        seg_df['Gap (%)'] = ((seg_df['AI Best'] - seg_df['Control']) / seg_df['Control'] * 100).round(1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Control',
            x=seg_df['Segment'],
            y=seg_df['Control'],
            marker_color='#2ecc71',
            text=seg_df['Control'].round(3),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Best AI',
            x=seg_df['Segment'],
            y=seg_df['AI Best'],
            marker_color='#f39c12',
            text=seg_df['AI Best'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            yaxis_title="HitRate@10",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Segment details
        st.markdown("### Segment Analysis")
        
        for _, row in seg_df.iterrows():
            with st.expander(f"📊 {row['Segment']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Control", f"{row['Control']:.3f}")
                with col2:
                    st.metric("Best AI", f"{row['AI Best']:.3f}")
                with col3:
                    st.metric("Gap", f"{row['Gap (%)']}%", delta_color="inverse")
        
        st.error("""
        **Finding:** Even power users (351 avg ratings) prefer genre-based recommendations
        - Heavy raters: Control 0.750 vs AI 0.562 (-25%)
        - Genre preferences dominate regardless of user sophistication
        """)
    
    else:  # Movie Characteristics
        st.subheader("🎬 Performance by Movie Type")
        
        st.markdown("""
        **Hypothesis:** AI might win for niche or new releases
        
        **Result:** Control wins for popular movies (where it matters). AI only wins for niche content with ~0% hit rates.
        """)
        
        movie_df = pd.DataFrame({
            'Category': ['New & Popular', 'Classic & Popular', 'New & Niche', 'Classic & Niche'],
            'Control': [0.066, 0.055, 0.000, 0.000],
            'AI Best': [0.023, 0.039, 0.0001, 0.005],
            'Hit Rate': ['6.6%', '5.5%', '~0%', '~0%']
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Control',
            x=movie_df['Category'],
            y=movie_df['Control'],
            marker_color='#2ecc71',
            text=movie_df['Hit Rate'],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='AI Best',
            x=movie_df['Category'],
            y=movie_df['AI Best'],
            marker_color='#f39c12'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            yaxis_title="Hit Rate",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.warning("""
        **Finding:** AI "wins" only for niche movies with negligible impact
        - Control design limits to popular movies (by design)
        - AI slightly better for long-tail (0.5% vs 0%)
        - But niche recommendations have ~0% engagement anyway
        """)
    
    # Summary
    st.markdown("---")
    st.subheader("🎯 Conditional Analysis Summary")
    
    summary_df = pd.DataFrame({
        'Dimension': ['Temporal Patterns', 'User Segments', 'Movie Types (Popular)'],
        'Conditions Tested': ['6 (weekday/weekend, 4 seasons)', '4 (casual, active, heavy, minimal)', '4 (new/old, popular/niche)'],
        'AI Wins': ['0 / 6', '0 / 4', '0 / 2 (popular categories)'],
        'Avg Gap': ['-25%', '-24%', '-48%']
    })
    
    st.dataframe(summary_df, width='stretch')
    
    st.error("""
    **Conclusion:** No conditional advantage found for AI models
    - Tested 13 distinct conditions across 3 dimensions
    - Control wins in every meaningful scenario
    - Only "wins" for AI are edge cases with ~0% hit rates
    """)

# ============================================================
# PAGE 5: BUSINESS RECOMMENDATIONS
# ============================================================
elif page == "💼 Business Recommendations":
    st.title("💼 Business Recommendations")
    
    # Main recommendation
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
        <h2 style='color: white; margin: 0;'>✅ RECOMMENDATION</h2>
        <h1 style='font-size: 2.5rem; margin: 1rem 0; color: white;'>
            Deploy the Control Baseline
        </h1>
        <p style='font-size: 1.2rem; margin: 0;'>
            Simple genre-based popularity recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##")
    
    # Rationale
    st.subheader("📋 Rationale")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Why Control Wins
        
        **Performance:**
        - 28% better engagement than best AI
        - Consistent across all user segments
        - Robust across temporal patterns
        
        **Operational:**
        - <1ms inference time (vs 28s-21min)
        - Zero training cost
        - No model maintenance required
        
        **Product:**
        - Interpretable recommendations
        - Easy to debug and explain
        - No "black box" issues
        """)
    
    with col2:
        st.markdown("""
        ### ❌ Why AI Falls Short
        
        **Metric Mismatch:**
        - Optimizes rating prediction (RMSE)
        - Business needs engagement (clicks)
        - 0.88 RMSE → 0.26 HitRate
        
        **Signal Already Captured:**
        - Genre preferences dominate
        - Control uses this optimally
        - AI adds no new information
        
        **Cost Not Justified:**
        - 1000x slower inference
        - Requires ongoing retraining
        - Worse results than free baseline
        """)
    
    st.markdown("---")
    
    # When to revisit
    st.subheader("🔄 When to Revisit AI")
    
    st.markdown("""
    Consider AI models if/when these conditions change:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **New Data Sources:**
        - Implicit signals available (clicks, watch time)
        - User demographic data
        - Social/collaborative features
        - Content metadata (actors, directors, plot)
        """)
        
        st.info("""
        **Business Goal Shifts:**
        - Diversity/serendipity prioritized over engagement
        - Cold-start problem becomes critical
        - Long-tail content needs promotion
        """)
    
    with col2:
        st.info("""
        **Technical Changes:**
        - Computational cost becomes negligible
        - Real-time training becomes feasible
        - Hybrid approaches can be tested easily
        """)
        
        st.info("""
        **Market Changes:**
        - User behavior shifts significantly
        - Genre preferences become less predictive
        - Competitive pressure requires sophistication
        """)
    
    st.markdown("---")
    
    # Implementation roadmap
    st.subheader("🗺️ Implementation Roadmap")
    
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
            'Metrics stable/improving',
            'Incremental improvements',
            'AI beats Control by >10%'
        ]
    })
    
    st.dataframe(roadmap, width='stretch', height=200)
    
    st.markdown("---")
    
    # Key lessons
    st.subheader("🎓 Key Lessons for ML Practitioners")
    
    st.success("""
    **When Simple Baselines Win:**
    1. **Domain signal is clear** → Genre preferences are explicit and dominant
    2. **Metric mismatch** → AI optimizes wrong objective (rating accuracy ≠ engagement)
    3. **Cost doesn't justify benefit** → Complexity adds latency without performance gain
    """)
    
    st.warning("""
    **Best Practices Demonstrated:**
    - ✅ Start with strong baselines before complex models
    - ✅ Use business-relevant metrics (engagement, not just accuracy)
    - ✅ Conduct conditional analysis (when does each approach win?)
    - ✅ Consider computational cost vs. marginal gains
    - ✅ Be willing to recommend against AI when appropriate
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Data: MovieLens 1M | Author: Rahul Yatham*")
# AI Personalization Analysis: When Simple Baselines Beat Machine Learning

A rigorous empirical study demonstrating that well-designed heuristics can outperform sophisticated AI models for recommendation systems, with implications for production ML deployment strategies.

---

## 🎯 Executive Summary

**Finding:** A simple genre-based popularity heuristic achieves **28% better engagement** than optimized AI models (SVD, KNN) with extensive feature engineering.

**Key Insight:** AI models optimized for rating prediction (RMSE) fail to capture the dominant signal for engagement: genre preferences combined with popularity.

**Business Impact:** Avoiding unnecessary AI complexity saves computational costs while delivering superior user experience.

---

## 📊 Results at a Glance

| Model | HitRate@10 | MRR@10 | Training Time | Inference Time |
|-------|------------|--------|---------------|----------------|
| **Control (Genre-based)** | **0.420** | **0.187** | 0s | <1ms |
| SVD (Pure) | 0.263 | 0.112 | 45s | 28s |
| SVD + Features | 0.302 | 0.106 | 45s | 28s |
| KNN (Pure) | 0.269 | 0.098 | 382s | 21min |
| KNN + Features | 0.312 | 0.104 | 382s | 21min |

**Control wins by 26-28% across ALL conditions tested:**
- ✅ Temporal: Weekdays, weekends, all seasons
- ✅ User segments: Light users, power users, all activity levels
- ✅ Movie types: New releases, classics, popular titles

---

## 🔬 Methodology

### Experiment 1: Baseline Model Comparison
- **Dataset:** MovieLens 1M (1M ratings, 6K users, 4K movies)
- **Split:** 60% train / 20% validation / 20% test (temporal)
- **Models:** 
  - Control: Popular movies within user's preferred genres
  - SVD: Matrix factorization (hyperparameter tuned)
  - KNN: Collaborative filtering (hyperparameter tuned)
- **Metrics:** HitRate@10, MRR@10 (engagement-focused)
- **Segmentation:** K-Means clustering (4 user segments)

### Experiment 2: Feature Engineering & Conditional Analysis
- **Features Added:**
  - Temporal: Movie release year, season, day of week, movie age at rating
  - User behavioral: Rating velocity, genre diversity, recency preferences
  - Interaction: Genre match scores (vectorized via matrix multiplication)
- **Hybrid Models:** Re-ranking with genre + temporal features (α=0.0 to 1.0)
- **Conditional Tests:**
  - Temporal patterns (weekday/weekend, seasons)
  - User segments (casual, active, heavy, minimal)
  - Movie characteristics (new/old, popular/niche)

---

## 💡 Key Findings

### 1. Genre Preferences Dominate Engagement
Users consistently engage with movies in their preferred genres, regardless of collaborative filtering scores. Simple genre-matching captures this better than latent factor models.

### 2. Metric Mismatch Problem
- **AI optimizes:** Rating prediction accuracy (RMSE)
- **Business needs:** Engagement (HitRate, click-through)
- **Result:** 0.88 RMSE (good) → 0.26 HitRate (poor)

### 3. Feature Engineering Helps, But Not Enough
- SVD: +15% improvement (0.263 → 0.302)
- KNN: +16% improvement (0.269 → 0.312)
- **Still loses by 26-28%** because baseline already uses genres optimally

### 4. No Conditional Advantage for AI
Tested temporal, user, and movie characteristics. Control wins everywhere meaningful. Only "wins" for AI were edge cases with ~0% hit rates (niche movies).

### 5. Computational Cost vs. Benefit
- Control: <1ms inference, no training cost
- AI: 28s-21min inference, requires retraining
- **Conclusion:** Cost not justified by performance

---

## 🛠️ Technical Implementation

### Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd ai-personalization-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Experiment 1: Baseline comparison
jupyter notebook notebooks/personalization_analysis_experiment#1.ipynb

# Experiment 2: Feature engineering & conditional analysis
jupyter notebook notebooks/personalization_analysis_experiment#2.ipynb
```

### Key Technologies
- **Data Processing:** Pandas, NumPy (vectorized operations for 23M+ precomputed scores)
- **ML Models:** scikit-surprise (SVD, KNN), scikit-learn (K-Means, StandardScaler)
- **Visualization:** Matplotlib, Seaborn
- **Optimization:** Numpy matrix operations, batch predictions, precomputed similarity matrices

---

## 📁 Project Structure
```
ai-personalization-analysis/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── data/
│   └── ml-1m/                                  # MovieLens 1M dataset
│       ├── ratings.dat
│       ├── movies.dat
│       └── users.dat
├── notebooks/
│   ├── personalization_analysis_experiment#1.ipynb  # Baseline models
│   └── personalization_analysis_experiment#2.ipynb  # Feature engineering
└── visualizations/
    ├── model_performance_comparison.png
    ├── experiment2_summary.png
    └── conditional_analysis.png
```

---

## 🎓 Lessons for ML Practitioners

### When Simple Baselines Win:
1. **Domain signal is clear:** Genre preferences are explicit and dominant
2. **Metric mismatch:** AI optimizes wrong objective (rating accuracy ≠ engagement)
3. **Cost doesn't justify benefit:** Complexity adds latency without performance gain

### When to Revisit AI:
- Implicit signals available (clicks, watch time, not just ratings)
- Business goal shifts to diversity/serendipity over pure engagement
- Cold-start becomes critical (AI slightly better for niche content)
- Computational cost becomes negligible

### Best Practices Demonstrated:
- ✅ Proper temporal train/val/test splits
- ✅ Hyperparameter tuning via grid search
- ✅ Business-relevant metrics (engagement, not just accuracy)
- ✅ Conditional analysis (when does each approach win?)
- ✅ Segmentation analysis (which users benefit?)
- ✅ Computational cost considerations
- ✅ Vectorized operations for scalability

---

## 📚 References

### Research Alignment
This analysis aligns with published findings:
- [Dacrema et al. (2019)](https://arxiv.org/abs/1907.06902): "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches" - Found 6 of 7 reproducible deep learning models could be outperformed by simple heuristics
- MovieLens benchmark studies consistently show genre-based filtering performs competitively with collaborative filtering

### Dataset
- **Source:** [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- **Citation:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)

---

## 👤 Author

**Your Name**  
*Data Scientist | Machine Learning Engineer*

[LinkedIn](your-linkedin) | [GitHub](your-github) | [Portfolio](your-portfolio)

---

## 📄 License

This project uses the MovieLens 1M dataset, which is provided for research purposes. Please cite the dataset appropriately if used in publications.

---

## 🙏 Acknowledgments

- GroupLens Research at the University of Minnesota for the MovieLens dataset
- scikit-surprise library for collaborative filtering implementations
- The ML research community for emphasizing the importance of strong baselines

---

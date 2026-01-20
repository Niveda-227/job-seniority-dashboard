# 🎯 Job Seniority Prediction Dashboard

Interactive Streamlit dashboard for predicting job seniority levels from posting content using machine learning.

**🌐 Live Demo:** [Launch Dashboard](YOUR-STREAMLIT-URL-HERE)

---

## 📊 Features

### 🏠 Project Overview
- Key metrics and project summary
- Model comparison visualization
- Tech stack and business impact

### 📈 Model Performance
- Detailed performance metrics for 4 algorithms
- Interactive comparison charts
- Accuracy, precision, recall, F1-score analysis

### 🔮 Live Prediction
- **Interactive predictor** - Try the model yourself!
- Input job details (title, skills, experience, salary)
- Get instant seniority predictions with confidence scores
- Visual probability breakdown

### 💡 Data Insights
- Key findings and learnings
- Feature importance analysis
- Dataset characteristics
- Class distribution visualization

---

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **Visualization:** Plotly
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Models:** Random Forest, Gradient Boosting, SVM, Neural Network

---

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/Niveda-227/job-seniority-dashboard.git
cd job-seniority-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
job-seniority-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── random_forest_model.pkl         # Random Forest classifier
├── gradient_boosting_model.pkl     # Gradient Boosting classifier
├── svm_model.pkl                   # SVM classifier
├── neural_network_model.pkl        # Neural Network classifier
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── label_encoder.pkl               # Label encoder
├── pca_transformer.pkl             # PCA transformer
├── scaler_pca.pkl                  # Scaler for PCA
└── scaler_svm_nn.pkl               # Scaler for SVM/NN
```

---

## 🎯 Model Performance

| Model | Test Accuracy | F1-Score |
|-------|--------------|----------|
| **Random Forest** | **74%** | **0.73** |
| Gradient Boosting | 73% | 0.72 |
| Neural Network | 72% | 0.71 |
| SVM | 70% | 0.69 |

All models significantly outperform the 56% baseline (majority class prediction).

---

## 📊 Dataset

- **Size:** 3,794 job postings (2024-2025)
- **Sources:** Multiple job boards (Indeed, LinkedIn, AI/ML job sites)
- **Classes:** Entry, Mid, Senior/Lead, Other
- **Features:** 60 engineered features → 47 PCA components

---

## 💡 Key Insights

1. **Job titles alone are insufficient** - Same title means different seniority across companies
2. **Multi-signal approach works best** - Combining title + skills + experience keywords
3. **Class imbalance is real** - Senior/Lead positions underrepresented in data
4. **Experience keywords are powerful** - Mentions of "3-5 years" or "5+ years" highly predictive

---

## 🔗 Related Projects

- **Main Analysis:** [Job Seniority Prediction ML Pipeline](https://github.com/Niveda-227/job-seniority-prediction)
- **Portfolio:** [Your Portfolio URL]

---

## 👤 Author

**Niveda Jawahar**  
Master's in Data Science | University of Maryland, College Park

- **LinkedIn:** [linkedin.com/in/nivedajawahar](https://www.linkedin.com/in/nivedajawahar/)
- **GitHub:** [github.com/Niveda-227](https://github.com/Niveda-227)
- **Email:** nivedajawahar@gmail.com

---

## 📝 License

This project is part of an academic portfolio. Feel free to explore and learn from the code!

---

## 🙏 Acknowledgments

- Dataset sources: Kaggle job posting datasets
- Built with Streamlit's amazing framework
- Deployed on Streamlit Cloud

---

*Last Updated: January 2026*

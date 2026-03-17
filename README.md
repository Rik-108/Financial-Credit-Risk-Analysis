# Financial Analytics: Predictive Modeling for Credit Risk 💳📊

## 1. Project Overview
This project is a comprehensive case study in **Financial Risk Analytics**. The objective is to build a predictive model that can identify high-risk financial profiles. This is a classic "Needle in a Haystack" problem where the goal is to accurately flag the small percentage of "Default" cases without over-penalizing "Safe" customers.

## 2. The Challenge: Class Imbalance
The primary technical hurdle in this dataset was the significant class imbalance (~94% vs 6%). 
- **The Pitfall:** A standard model could achieve 94% accuracy by simply guessing "Safe" every time, but it would fail to detect any "Default" events.
- **The Solution:** Implemented **Balanced Resampling** techniques during the training phase to ensure the models learned the characteristics of the minority class.

## 3. Model Comparison & Evaluation
I evaluated multiple machine learning architectures to find the optimal balance between **Precision** and **Recall**:

* **Logistic Regression:** Used as a baseline for interpretability.
* **Support Vector Machines (SVM):** Utilized for finding the optimal hyperplane in high-dimensional financial feature spaces.
* **Gaussian Naive Bayes:** Tested for its efficiency in handling probabilistic financial indicators.

### Performance Summary (Naive Bayes Example):
- **Accuracy:** 94.35%
- **Recall (Class 1):** 38% (Demonstrating the model's ability to catch defaults that standard models miss).

## 4. Key Analytics Workflow
1.  **Data Preprocessing:** Feature scaling using `MinMaxScaler` to ensure numerical stability across different financial scales.
2.  **Imbalance Handling:** Applied `resample` logic to prevent model bias.
3.  **Visual Evaluation:** Generated **Confusion Matrices** and **Classification Reports** to provide a transparent view of model trade-offs beyond simple accuracy.

## 5. Technology Stack
- **Languages:** Python
- **Libraries:** Scikit-Learn (Preprocessing, Modeling, Metrics), Pandas, Seaborn, Matplotlib.
- **Concepts:** Supervised Learning, Risk Scoring, Data Resampling.

## 6. How to Use
1.  **Clone the Repo:** `git clone https://github.com/yourusername/Financial-Credit-Risk-Analysis.git`
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Run Analysis:** Open `notebooks/credit_risk_modeling.ipynb` to view the full pipeline.

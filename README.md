# GDP Prediction using Nighttime Light Data

## Overview
This project explores the use of satellite-based **Nighttime Light (NTL) data** to predict GDP across regions. The goal is to leverage machine learning models to establish a correlation between NTL intensity and economic activity, providing an alternative GDP estimation approach, especially for regions with limited economic data.

## Methodology
- **Data Collection:** Processed **VIIRS satellite imagery** for Nighttime Light (NTL) data and integrated it with economic indicators such as **Sensex, repo rate, and exchange rate**.
- **Feature Engineering:** Extracted insights from NTL data and engineered economic features to improve model performance.
- **Modeling Approaches:** Applied multiple regression and machine learning models, including **Random Forest, XGBoost, ANN, and CNN**.
- **Performance Optimization:** Conducted **feature importance analysis**, hyperparameter tuning via **grid search**, and **cross-validation** to enhance model accuracy.

## Key Findings
- Achieved **95% fitness** in GDP prediction.
- Model **outperformed traditional methods by 22%**, demonstrating the effectiveness of satellite-based GDP estimation.
- The **best model (R-squared: 0.89, RMSE: 0.15)** showed high predictive accuracy.

## Conclusion
The study confirms that **Nighttime Light (NTL) data is a reliable proxy for economic activity** and can significantly improve GDP estimation in data-scarce regions. The approach provides a scalable and efficient solution for economic analysis using remote sensing and AI.

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/gdp-prediction-ntl.git
   cd gdp-prediction-ntl
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```sh
   python main.py
   ```

## Technologies Used
- **Python** (NumPy, Pandas, Scikit-learn, TensorFlow, XGBoost)
- **Satellite Data Processing** (VIIRS Nighttime Light Data)
- **Machine Learning & Deep Learning**

## Future Scope
- Expanding the dataset with additional economic indicators.
- Exploring deep learning techniques like **transformers for time-series forecasting**.
- Enhancing interpretability using **SHAP (SHapley Additive exPlanations)**.

## Author
**Sanket Jagtap**  
IIT Kharagpur - Agricultural & Food Engineering

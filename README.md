# **House Price Prediction Model**

## **Overview**

I build the **House Price Prediction Model** machine learning project aimed at accurately predicting house prices based on essential real estate features with real world data. Leveraging advanced regression techniques and a dataset from 7 Indian cities, this project demonstrates end-to-end implementation, from data preprocessing to deployment using Flask and Docker.

By analyzing and comparing multiple models, optimizing hyperparameters, and deploying the best-performing model, this project provides me the practical insights into building and deploying machine learning models in the real estate domain with real world dataset.

---

## **Key Features**
- **End-to-End ML Pipeline**: Includes data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment.
- **Dataset Insights**: Real-world real estate data, addressing outliers, skewness, and missing values.
- **Model Evaluation**: Comparative analysis of regression algorithms.
- **Deployment Ready**: Flask API integrated with Docker for real-world use cases.

---

## **Dataset**
We used the dataset available at [Kaggle: Real Estate Data from 7 Indian Cities](https://www.kaggle.com/datasets/rakkesharv/real-estate-data-from-7-indian-cities?select=Real+Estate+Data+V21.csv). 

### **Dataset Details**
- **File Name**: `house_prices.csv`, Total 10.4mb dataset. for training and testing i have used 80/20 rule.
- Training set size: 11369
- Testing set size: 2843
- **Features**:
  - `Total Area`: The area of the property in square feet.
  - `Price per Square Foot`: The cost of the property per square foot.
  - `Baths`: Number of bathrooms in the property.
  - **Other Features Ignored**: Due to irrelevance or redundancy in prediction tasks (e.g., IDs, textual data).
- **Shape**: 14,528 rows × 3 columns (post-cleaning).

### **Challenges in Dataset**:
1. **Outliers**: Extreme values in `Total Area` and `Price per Square Foot`.
   - Example: Properties with unusually high or low prices skewed the data distribution.
2. **Missing Values**:
   - Missing entries for `Price per Square Foot` and `Baths` were filled using median imputation.
3. **Skewed Distribution**:
   - Log transformation was applied to normalize highly skewed features.

**Preprocessing**:
- Feature Scaling: Used `StandardScaler` for consistent model input.
- Outlier Handling: Winsorized outliers to reduce their impact without completely removing them.

---

## **Models Used and Evaluation**
We evaluated four models to identify the best fit for predicting house prices.

### **1. Linear Regression**
- **Results**:
  - MAE: 0.6259
  - MSE: 1.0027
  - RMSE: 1.0014
- **Analysis**:
  - Linear Regression served as a baseline.
  - It underperformed on non-linear relationships within the data.

### **2. Random Forest Regressor**
- **Results**:
  - MAE: 0.6406
  - MSE: 1.0681
  - RMSE: 1.0335
- **Analysis**:
  - Captures non-linear relationships better than Linear Regression.
  - Prone to overfitting on smaller datasets.

### **3. Gradient Boosting Regressor**
- **Results**:
  - MAE: 0.6117
  - MSE: 0.9666
  - RMSE: 0.9832
- **Analysis**:
  - Achieved superior performance by minimizing errors.
  - Computationally expensive during hyperparameter tuning.

### **4. Optimized Gradient Boosting Regressor**
- **Results**:
  - MAE: 0.6119
  - MSE: 0.9641
  - RMSE: 0.9819
- **Optimizations**:
  - `learning_rate=0.05`, `n_estimators=200`, `max_depth=3`.
  - Cross-validation ensured generalization to unseen data.
- **Analysis**:
  - Demonstrates the importance of tuning hyperparameters.
  - Selected for deployment due to its balance of performance and efficiency.

---

## **Expert Insights**
1. **Why Gradient Boosting Works Best**:
   - Combines weak learners iteratively to reduce prediction error.
   - Handles non-linear data effectively while controlling overfitting through regularization.
   - Robust to small dataset size compared to Random Forest.

2. **Deployment Decisions**:
   - Pickled the trained model (`house_price_predictor_model.pkl`) and the scaler (`scaler.pkl`) for efficient loading during predictions.
   - Chose Flask for a lightweight, fast API setup.

3. **Possible Improvements**:
   - Incorporate location data and advanced feature engineering (e.g., distance to city center).
   - Use ensemble techniques like stacking for enhanced predictions.

---

## **Deployment**
The project includes a Flask API for serving predictions and is containerized using Docker for scalability.

### **Folder Structure**

house-price-prediction/
notebooks/
house_price_prediction.ipynb  # Jupyter Notebook for EDA and model development

data/
house_prices.csv              # Cleaned dataset

src/
app.py                        # Flask API for predictions
house_price_predictor.pkl     # Trained Gradient Boosting Model
scaler.pkl                    # Standard Scaler object

requirements.txt                  # Python dependencies
Dockerfile                        # Docker configuration
README.md                         # Project documentation

### **Steps to Run Locally**
1. Clone the repository:
   git clone https://github.com/ManishSnowflakes/house-price-prediction.git
   cd house-price-prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Start the Flask API:
   python src/app.py

4. Build and Run Docker Image:
   docker build -t house-price-prediction .
   docker run -p 5000:5000 house-price-prediction

5. Access the API at `http://localhost:5005`.

---

## **API Usage**
- **POST /predict**
  - Request Body:
    ```json
    {
      "Total_Area": 1200,
      "Price_per_SQFT": 6000,
      "Baths": 2
    }
    ```
  - Response:
    ```json
    {
      "predicted_price": 7.2
    }
    ```

---

## **Performance Analysis**
- **MAE**: Indicates average absolute error in predictions (lower is better).
- **MSE and RMSE**: Highlight the variance and standard error in predictions.
- **Conclusion**:
  - The model's RMSE (~0.98) demonstrates high accuracy with minimal prediction errors.
  - Gradient Boosting outperformed all other models, validating its robustness for this dataset.

---

## **Future Enhancements**
1. **Incorporate More Features**:
   - Location, property type, and age of property can significantly enhance accuracy.
2. **Improve Scalability**:
   - Deploy on cloud platforms like AWS or GCP for handling high traffic.
3. **Enhance API**:
   - Add detailed error messages and input validation for production readiness.
4. **Interactive Dashboard**:
   - Build a React or Angular-based frontend for user-friendly interactions.

---

## **Acknowledgements**
I thank [Kaggle](https://www.kaggle.com/) for providing the dataset and the open-source community for supporting development tools.

**For inquiries and contributions, please contact [Manishsnowflakes@gmail.com] or raise a GitHub issue.**

---

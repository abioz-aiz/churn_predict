# Stock Price & Churn Prediction Application

[![Demo Video](https://github.com/abioz-aiz/churn_predict/blob/main/Screenshot%202024-11-29%20144319.png)](https://youtu.be/kN9NdZ2h2w8)

(Click on the Image to be redirected to the video)

This repository contains two main components:
1. A Stock Price visualization app using Streamlit
2. A Customer Churn Prediction model using Flask

## Project Structure

project/

├── churn_predict/

│ ├── datasets/

│ │ └── churn_data.csv

│ ├── model/

│ │ └── model_C=1.0.bin

│ ├── deploy_churn.ipynb

│ ├── myapp.py

│ ├── Dockerfile

│ └── requirements.txt

└── README.md


## Stock Price Visualization App

### Features
- Displays Google (GOOGL) stock closing prices
- Shows trading volume visualization
- Interactive date range selection
- Built with Streamlit for easy web interface

### Requirements

python
streamlit==1.31.0
yfinance==0.2.36
pandas==2.1.0


### Running the Stock Price App

   streamlit run myapp.py

## Churn Prediction Model

### Features
- Predicts customer churn probability
- Uses Logistic Regression model
- ROC AUC score of ~0.858 on test set
- REST API endpoint for predictions

### Model Details
- Training data includes customer demographics and service usage
- Features include tenure, monthly charges, total charges, and various categorical variables
- Model trained using scikit-learn's LogisticRegression with 5-fold cross-validation

### Requirements

python
flask==3.0.0
pandas==2.1.0
scikit-learn==1.3.0
gunicorn==21.2.0
numpy==1.24.0

### Running Locally

1. **Install dependencies**
   pip install -r requirements.txt
2. **Run Flask application**
   python predict.py
3. **Making Predictions**
   Test the API using curl:
   bash
   
curl -X POST -H "Content-Type: application/json" -d '{

"gender": "female",

"seniorcitizen": 0,

"partner": "yes",

"dependents": "no",

"phoneservice": "no",

"multiplelines": "no_phone_service",

"internetservice": "dsl",

"onlinesecurity": "no",

"onlinebackup": "yes",

"deviceprotection": "no",

"techsupport": "no",

"streamingtv": "no",

"streamingmovies": "no",

"contract": "month-to-month",

"paperlessbilling": "yes",

"paymentmethod": "electronic_check",

"tenure": 1,

"monthlycharges": 29.85,

"totalcharges": 29.85

}' http://localhost:9696/predict

5. Running the Streamlit App
   streamlit run predict_copy.py

## Development

The project uses:
- Python 3.8+
- Jupyter Notebook for model development
- Flask for API development
- Docker for containerization
- Streamlit for stock visualization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source for stock prices: Yahoo Finance
- Churn prediction dataset: [Churn_data.csv](https://github.com/abioz-aiz/churn_predict/blob/main/datasets/churn_data.csv)

Contact: 

[Zoiba Zia](https://www.linkedin.com/in/zoiba/)

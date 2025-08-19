# Stock Market Forecasting using Machine Learning

This project demonstrates two distinct machine learning approaches for analyzing and forecasting stock market trends: predicting future stock prices using a Long Short-Term Memory (LSTM) neural network and forecasting stock price movement (up/down) using a Random Forest classifier. Both approaches leverage historical stock data and incorporate various data preprocessing and feature engineering techniques.

## Features

*   **Historical Stock Data Acquisition**: Utilizes the `yfinance` library to download historical stock data for specified tickers and date ranges.
*   **LSTM Model for Price Prediction**:
    *   Implements a multi-layered Long Short-Term Memory (LSTM) neural network to predict future stock closing prices.
    *   Includes data scaling using `MinMaxScaler` for optimal model performance.
    *   Generates sequential data (time-series) suitable for LSTM input.
    *   Visualizes predicted prices against actual prices to assess model accuracy.
    *   Saves the trained LSTM model in Keras format.
    *   Integrates with Hugging Face Hub for model sharing and deployment.
*   **Random Forest Model for Trend Prediction**:
    *   Employs a `RandomForestClassifier` to predict binary stock price movement (up/down) for the next trading day.
    *   Performs extensive feature engineering, including:
        *   Calculation of rolling averages (`Close_Ratio_X`) for various time horizons (e.g., 2, 5, 60, 250, 1000 days).
        *   Computation of past trend sums (`Trend_X`) to capture historical up/down movements.
    *   Implements a robust backtesting strategy to evaluate the model's performance over time, simulating real-world trading scenarios.
    *   Evaluates classification performance using metrics such as precision, accuracy, recall, and F1-score.
    *   Visualizes predicted stock movements (buy/sell signals) overlaid on historical stock prices.
*   **Data Preprocessing**: Handles essential data cleaning steps, such as dropping missing values (`dropna()`) and creating lagged features for time-series analysis.
*   **Model Training & Evaluation**: Provides code for training, compiling, and evaluating the machine learning models.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository** (if you have the original GitHub repo) or create the `.ipynb` files locally with the provided content.

    ```bash
    git clone https://github.com/galang006/forecasting_stock_market.git
    cd forecasting_stock_market
    ```

2.  **Install the required Python libraries**. You can install them using pip:

    ```bash
    pip install pandas numpy matplotlib yfinance scikit-learn tensorflow huggingface_hub
    ```

    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical operations.
    *   `matplotlib`: For data visualization.
    *   `yfinance`: To download historical stock data.
    *   `scikit-learn`: For machine learning models (Random Forest, MinMaxScaler) and evaluation metrics.
    *   `tensorflow`: For building and training the LSTM neural network (Keras is part of TensorFlow).
    *   `huggingface_hub`: For interacting with Hugging Face Hub (used for model upload in `stock_market_predict_LSTM.ipynb`).

3.  **(Optional) Set up Hugging Face Token**: If you plan to upload the trained LSTM model to Hugging Face Hub, you will need an authentication token.
    *   Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to create a new token with 'write' access.
    *   You can log in via the command line within your notebook environment using:
        ```bash
        !huggingface-cli login
        ```
        Follow the prompts to enter your token.

## Usage

The project consists of three Jupyter notebooks, each serving a distinct purpose in the stock market forecasting pipeline.

### 1. `scraping_stock_market_data.ipynb`

This notebook is used to download historical stock data and save it locally.

*   **Purpose**: Scrape historical stock data for a specific ticker (`bbri.jk` by default) from Yahoo Finance and save it as a CSV file.
*   **How to Run**:
    1.  Open the notebook in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab).
    2.  Execute all cells sequentially.
*   **Output**: A CSV file named `bbri_stock_data.csv` will be created in the same directory as the notebook, containing historical stock data. The head of the DataFrame will also be printed to the console.

### 2. `stock_market_predict_LSTM.ipynb`

This notebook demonstrates building and training an LSTM model for stock price prediction (regression task).

*   **Purpose**: Preprocess historical stock closing prices, train an LSTM model to predict future prices, and visualize the predictions.
*   **How to Run**:
    1.  Ensure `bbri_stock_data.csv` (or similar historical data) is available, or the notebook will download it directly using `yfinance`.
    2.  Open the notebook in a Jupyter environment.
    3.  Execute all cells sequentially.
    4.  **Note**: The notebook includes code to create a `model` directory and save the trained Keras model (`Stock Predictions Model.keras`) within it. It also attempts to upload this model to Hugging Face Hub, which requires a configured `HF_TOKEN`. If you don't have this, the upload cell might fail, but the model training and prediction will still complete.
*   **Output**:
    *   Plots showing the 100-day and 200-day moving averages overlaid on the stock's closing price.
    *   A plot comparing the predicted stock prices by the LSTM model against the original (actual) prices on the test set.
    *   A saved Keras model file: `model/Stock Predictions Model.keras`.

### 3. `stock_market_predict_random_forest.ipynb`

This notebook focuses on using a Random Forest classifier to predict stock price direction (up/down) and evaluates it using a backtesting strategy.

*   **Purpose**: Engineer features from historical stock data, train a Random Forest classifier to predict the next day's stock movement, and evaluate its performance using backtesting.
*   **How to Run**:
    1.  The notebook will download historical data for `bbri.jk` directly using `yfinance`.
    2.  Open the notebook in a Jupyter environment.
    3.  Execute all cells sequentially.
*   **Output**:
    *   Various DataFrame outputs displaying the engineered features.
    *   Performance metrics (precision, accuracy, recall, f1-score) of the Random Forest model on the backtested predictions.
    *   A visual plot overlaying the stock's closing price with markers indicating predicted "up" (green triangles) or "down" (red triangles) days.

## Code Structure

The project is organized as a collection of Jupyter notebooks:

```
forecasting_stock_market/
├── scraping_stock_market_data.ipynb
├── stock_market_predict_LSTM.ipynb
└── stock_market_predict_random_forest.ipynb
```

*   `scraping_stock_market_data.ipynb`: Contains code for downloading raw historical stock data and saving it.
*   `stock_market_predict_LSTM.ipynb`: Implements the LSTM neural network for stock price regression. Handles data preparation, model architecture, training, and result visualization.
*   `stock_market_predict_random_forest.ipynb`: Implements the Random Forest classifier for stock price trend prediction. Focuses on feature engineering, backtesting methodology, and classification metrics.
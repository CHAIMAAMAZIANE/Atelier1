# NYSE Stock Price Regression with PyTorch

This project explores historical NYSE stock-market data and builds a deep neural network to predict the next trading session's closing price. It was developed as a regression lab using Python, PyTorch, and scikit-learn.

> **Note:** Run the notebook from top to bottom to generate the latest metrics, plots, and `results.csv`. Numerical results are intentionally not hard-coded in this README because they should correspond to the final executed notebook.

## Project objectives

- Perform exploratory data analysis (EDA) on historical stock prices and related company data.
- Engineer features from price, volume, returns, moving averages, and recent volatility.
- Build a multi-layer perceptron (MLP) with PyTorch for a regression task.
- Tune hyperparameters with `GridSearchCV` and time-ordered cross-validation.
- Compare a baseline neural network with regularization techniques.
- Evaluate the model using regression metrics and directional accuracy.

## Dataset

The project uses the public [NYSE dataset on Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse), including files such as:

- `prices-split-adjusted.csv` — historical prices and trading volume; used for feature engineering and prediction.
- `securities.csv` — company and sector metadata; used in EDA.
- `fundamentals.csv` — financial fundamentals; used in EDA.

The notebook downloads the dataset with KaggleHub. The raw dataset is not included in this repository.

## Methodology

### 1. Exploratory data analysis

The notebook examines dataset dimensions, data types, descriptive statistics, missing values, duplicates, company coverage, price and volume distributions, outliers, correlations, historical price trends, daily returns, and sector-level information.

### 2. Feature engineering and target

Features are calculated per stock symbol and include:

- Log-transformed open, high, low, close, and volume values.
- Current and lagged log returns.
- Distance from 5-day and 10-day moving averages.
- 10-day rolling volatility.

The target is the **next observation's log closing price for the same stock symbol**. Rows with missing features or targets are removed. The data is then ordered chronologically.

### 3. Chronological split and scaling

The notebook uses historical observations before 2016 for model development and observations from 2016 for final testing. Within the pre-2016 data, observations before July 2015 are used for training and later observations for validation. Standardization is fitted on the training period and applied to validation and test data using the same fitted scalers.

### 4. Neural network and hyperparameter search

A configurable MLP is implemented in PyTorch. It uses fully connected layers, ReLU activations, and one output neuron for the continuous regression target. The notebook searches over learning rate, optimizer (Adam or RMSprop), number of epochs, and hidden-layer architecture using `GridSearchCV` with chronological splits. A 40,000-row subset is used for the grid search to reduce computation time.

### 5. Regularization experiments

The baseline model is compared with experiments using:

- Dropout (0.2)
- L2 regularization / weight decay
- Batch normalization
- Early stopping based on validation loss
- Combinations of these methods
- A lower learning rate

### 6. Evaluation

The notebook reports:

- **MSE loss** on standardized targets during training and validation.
- **R²** on the log closing-price target.
- **RMSE in dollars**, calculated after transforming log prices back to price scale.
- **Directional accuracy**, which checks whether the predicted price movement has the same direction as the observed movement.

A naive baseline that predicts tomorrow's close to be equal to today's close is included. This baseline is important because stock prices often change relatively little from one trading session to the next. A high R² alone does not prove that the neural network is useful; its results should be compared with this simple baseline.

## Repository contents

```text
.
├── NYSE_Deep_Learning_Regression.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

When the notebook is executed, it also creates an `images/` directory with plots and a `results.csv` file containing the final model comparison.

## How to run

### Recommended: Google Colab

1. Open the notebook in Google Colab.
2. Run all cells in order.
3. The notebook installs/downloads the required dataset dependencies and retrieves the NYSE dataset through KaggleHub.
4. Review the EDA plots, training/validation curves, and final model-comparison table.

### Local environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Start Jupyter and open `NYSE_Deep_Learning_Regression.ipynb`:

```bash
jupyter notebook
```

A CUDA-compatible GPU can speed up training, but the notebook can also run on CPU. Full grid search and repeated training may take time.

## Results and interpretation

After running the notebook, use its final comparison table and saved plots to report the observed metrics. Compare every regularized model with both the unregularized MLP and the naive baseline. Do not assume that adding regularization improves performance: the effect must be assessed from the validation and test results.

**Final findings:** _Complete this section after running the corrected notebook and checking the final `results.csv`._

## Limitations

- Historical price data does not capture every factor that influences market movements.
- The model is evaluated on a particular historical period; performance may differ in other market conditions.
- A good price-level R² can be misleading when the next price is close to the current price, which is why the naive baseline and directional accuracy are also reported.
- This project is an educational experiment, not financial advice or a trading system.

## Technologies

- Python, NumPy, pandas
- Matplotlib and Seaborn
- PyTorch
- scikit-learn and Skorch
- KaggleHub
- Jupyter Notebook / Google Colab

## Learning synthesis

This lab demonstrates the full workflow of a regression project: exploring and preparing data, engineering features, creating a PyTorch neural network, tuning hyperparameters, tracking learning curves, applying regularization, and evaluating against a simple baseline. An important lesson is that model complexity does not guarantee better generalization; reliable evaluation and meaningful baselines are essential.

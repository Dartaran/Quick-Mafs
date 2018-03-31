'''
File:
    custom_strategy.py

Description:
    Our portfolio construction strategy for the 2018 UChicago Midwest Trading
    Competition.

Authors:
    Alan Lu, Dylan Hong, Hanting Guo, and Shaket Chaudhary (Dartmouth College)
'''

from portfolio import PortfolioGenerator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from random import randint

class CustomStrategy(PortfolioGenerator):

    def __init__(self):
        pass

    def build_signal(self, stock_features):
        signal = pd.Series()

        # print(stock_features)

        # get the predicted return for this timestep to determine whether to buy or sell
        timestep = stock_features.index.max() + 1
        # iterate through each stock and normalize the predicted return to get portfolio weights from -100 to 100
        for i in range(0, 1000):
            predicted_stock_returns = predicted_signals[i]
            # predicted return for this timestep (not normalized)
            predicted_return = predicted_stock_returns[timestep]
            # print("Predicted return for stock " + str(i) + " at timestep " + str(timestep) + ": " + str(predicted_return))
            # normalize on a scale of -100 to 100
            normalized_return = 200 * ((predicted_return - predicted_returns_min) / (predicted_returns_max - predicted_returns_min)) - 100
            # append the normalized predicted return to the signal
            signal.set_value(i, normalized_return)

        print(signal)
        return signal

    def momentum(self, stock_features):
        return stock_features.groupby(['ticker'])['returns'].mean()


# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    portfolio = CustomStrategy()
    stock_df = portfolio.read_stock_data()
    print(stock_df)

    # calculate min and max returns for normalization on a scale of -100 to 100
    print(stock_df["returns"].mean())
    print(stock_df["returns"].max())
    print(stock_df["returns"].min())
    # returns_max = stock_df["returns"].max()
    # returns_min = stock_df["returns"].min()
    predicted_returns_min = 1000
    predicted_returns_max = -1000

    predicted_signals = []

    # regress each stock using gls
    for i in range(0, 1000):
        # grab each stock
        each_stock_df = stock_df.query("ticker == @i")
        # regress all the features on the stock using GLS
        result = smf.gls("returns ~ VIX + COPP + df + US_TRY + BIG_IX + SMALL_IX + SENTI + TEMP + RAIN + OIL",
                         each_stock_df).fit()
        # append the predicted returns for every timestep for this stock to an array
        predicted_signals.append(result.predict(each_stock_df))
        if result.predict(each_stock_df).min() < predicted_returns_min:
            predicted_returns_min = result.predict(each_stock_df).min()
        if result.predict(each_stock_df).max() > predicted_returns_max:
            predicted_returns_max = result.predict(each_stock_df).max()
        # print(result.predict(each_stock_df))
        print("Predicting returns for ticker " + str(i) + "...")

    # print(predicted_signals)

    sharpe = portfolio.simulate_portfolio()
    print("*** Strategy Sharpe is {} ***".format(sharpe))


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


# x = PortfolioGenerator()
# stock_df = x.read_stock_data()
# print(x.read_stock_data())
#
# print(stock_df["returns"].mean())
# print(stock_df["returns"].max())
# print(stock_df["returns"].min())

#
# for i in range(0, 999):
#     each_stock_df = stock_df.query("ticker == @i")
#     result = smf.gls("returns ~ VIX + COPP + df + US_TRY + BIG_IX + SMALL_IX + SENTI + TEMP + RAIN + OIL",
#                      each_stock_df).fit()
#     print(result.predict(each_stock_df))


# Forward selection stepwise regression
# data_df = stock_df[["VIX", "returns"]]
# result = forward_selected(data_df, "returns")
#
# print(result.summary())

# GLM regression
# data = stock_df
# glm_df = stock_df.copy()
# glm_df.drop("returns")
# data.endog = glm_df
# data.exog = stock_df["returns"]
#
# result = sm.GLM(data.endog, data.exog).fit()

# training_x = stock_df.copy()
# training_x.drop("returns")
# training_x.drop("industry")
# training_y = stock_df["returns"]
# reg = linear_model.LinearRegression()
# reg.fit(training_x, training_y)

# result = smf.ols("market_cap ~ VIX", data = stock_df).fit()

# print(result.summary())

# stock_df['3M_R'].type()

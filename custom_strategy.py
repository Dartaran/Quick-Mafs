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
import statsmodels.formula.api as smf

class CustomStrategy(PortfolioGenerator):

    def __init__(self):
        pass

    def build_signal(self, stock_features):
        return self.momentum(stock_features)

    def momentum(self, stock_features):
        return stock_features.groupby(['ticker'])['returns'].mean()

# Test out performance by running 'python sample_strategy.py'
# if __name__ == "__main__":
#     portfolio = SampleStrategy()
#     sharpe = portfolio.simulate_portfolio()
#     print("*** Strategy Sharpe is {} ***".format(sharpe))

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


x = PortfolioGenerator()
stock_df = x.read_stock_data()
print(x.read_stock_data())

result = forward_selected(stock_df, "returns")

print(result.summary())

# result = smf.ols("market_cap ~ VIX", data = stock_df).fit()

# print(result.summary())

# stock_df['3M_R'].type()

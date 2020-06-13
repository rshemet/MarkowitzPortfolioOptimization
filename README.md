# MarkowitzPortfolioOptimization
Computing a solution for the optimal mean-variance tradeoff (maximising Sharpe Ratio) of a portfolio according to MPT.

This repository contains code that allows you to extract the composition and performance 
of any exchange-traded fund and attempt to allocate its constituent assets differently,
according to the mean-variance optimization framework.

# Algorithms supported:

- Unconstrained optimization
- Constrained (sum weights = 1) optimization
- Short-selling constrained optimization

# Instructions

Data/ETF/ should have two .csv files for the benchmark fund: one with performance and one 
with composition \
Data/Stocks/ should have all the stock files downloaded from Kaggle

# Required libraries:

pandas
numpy
matplotlib
time
os
shutil
datetime
(pdb)
scipy
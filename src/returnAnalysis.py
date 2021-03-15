import pandas as pd
import numpy as np
import scipy.stats as st



# Calculate log return
# Input:
# - value_init:  Value at time T-1
# - value_final: Value at time T
# Output:
# - Log return value
#
def log_return(value_init, value_final):
    return np.log(value_final / value_init)


# Generate dataframe of log return
# Input:
# - df: Dataframe with assets values
# Output:
# - Dataframe with log return of assets
#
def generate_log_return(df):
    df_return = pd.DataFrame(columns=df.columns[1:])
    df_return['ibov'] = log_return(df.ibov.shift(periods=1, fill_value=df['ibov'].iloc[0]), df.ibov)
    df_return['tots3'] = log_return(df.tots3.shift(periods=1, fill_value=df['tots3'].iloc[0]), df.tots3)
    return df_return


# PDF of dataframe considering normal distribution
# Input:
# - df: Dataframe with log return of assets
# - ptrim: Percentile to ignore from the edges (default: 0.01)
# Output:
# - Probability Distribution Function of dataframe
#
def normal_simulation(df, ptrim=0.01):
    if df.ndim == 1:
        df_norm = st.norm(loc=np.mean(df), scale=np.std(df))
        dist = df_norm.pdf(np.linspace(st.norm.ppf(ptrim), st.norm.ppf(1-ptrim), 100))
    else:
        dist = df.apply(lambda x: normal_simulation(x, ptrim))
    return dist


# Confidence interval considering normal distribution
# Input:
# - df: Dataframe with log return of assets
# - alpha: significance level
# Output:
# - Plot with probability distribution function
#
def confidence_interval(df, alpha):
    return st.norm.interval(alpha=alpha, loc=np.mean(df), scale=np.std(df))


# Hypothesis Test: Mean return is equal some value?
# Input:
# - df: Dataframe with log return of assets
# - hyp_alt: Alternative hypothesis (value of the mean return)
# - alpha: significance level
# Output:
# - If alternative hypothesis is True or False
#
def hyp_return_is(df, hyp_alt, alpha):
    ci = confidence_interval(df, alpha)
    return ci[0] <= hyp_alt <= ci[1]



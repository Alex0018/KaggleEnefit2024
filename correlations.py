import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.styles import PALETTE_HEATMAP

def compute_correlations(X, Y):
    corrs = [(f, abs(X[f].corr(Y))) for f in X.columns]
    df_corrs = pd.DataFrame(corrs)
    df_corrs.index = df_corrs[0]
    df_corrs = df_corrs.drop(0, axis=1)
    df_corrs = df_corrs.reset_index()
    df_corrs.columns = ['feature', 'corr']

    return df_corrs


def compute_correlations_all(X, Y):
    df = compute_correlations(X, Y)
    for v in sorted(Y.unique()):
        df1 = compute_correlations(X, (Y == v).astype(int))
        df1.columns = ['feature', f'cor_{v}']
        df = pd.merge(df, df1, on='feature')

    df1 = compute_correlations(X, (Y < 2).astype(int))
    df1.columns = ['feature', 'cor_less_2']
    df = pd.merge(df, df1, on='feature')

    df1 = compute_correlations(X, (Y > 5).astype(int))
    df1.columns = ['feature', 'cor_greater_5']
    df = pd.merge(df, df1, on='feature')
    
    return df


def display_cormatrix(cormatrix):
    mask = np.eye(len(cormatrix)) # mask out identity correlations    
    
    plt.figure(figsize=(cormatrix.shape[1] * 1.1, cormatrix.shape[1] * 0.6))
    
    heatmap = sns.heatmap(cormatrix, mask=mask, vmin=-1, vmax=1, annot_kws={'alpha': 0.5},
                          annot=True, cmap=sns.color_palette(PALETTE_HEATMAP), cbar=False)
    heatmap.set_title('Feature correlations');
    
    for label in heatmap.get_xticklabels():
        label.set_rotation(90)
    for label in heatmap.get_yticklabels():
        label.set_rotation(0)
    
    plt.tight_layout()
    plt.show()
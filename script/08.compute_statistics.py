import pandas as pd
from math import sqrt
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon


def cohend(d1, d2):
    """
    function to calculate Cohen's d for independent samples
    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    d = (u1 - u2) / s
    d = abs(d)

    result = ''
    if d < 0.2:
        result = 'negligible'
    if 0.2 <= d < 0.5:
        result = 'small'
    if 0.5 <= d < 0.8:
        result = 'medium'
    if d >= 0.8:
        result = 'large'

    return result, d


if __name__ == '__main__':
    df = pd.read_excel('..\\app_results\\app_comparison_baseline.xlsx', engine='openpyxl')
    # df = df.iloc[:, :-2]
    # print(df.head())
    f_clone_DT = list(df[df['Classifier'] == 'Decision Tree']['f1 pos_clone'])
    f_distinct_DT = list(df[df['Classifier'] == 'Decision Tree']['f1 pos_distinct'])

    f_clone_PD = list(df[df['Classifier'] == 'VISUAL_PDiff']['f1 pos_clone'])
    f_distinct_PD = list(df[df['Classifier'] == 'VISUAL_PDiff']['f1 pos_distinct'])

    f_clone_DOM = list(df[df['Classifier'] == 'DOM_RTED']['f1 pos_clone'])
    f_distinct_DOM = list(df[df['Classifier'] == 'DOM_RTED']['f1 pos_distinct'])

    # compute wilcoxon test
    w1, p1 = wilcoxon(f_clone_DT, f_clone_PD)
    w2, p2 = wilcoxon(f_distinct_DT, f_distinct_PD)
    w3, p3 = wilcoxon(f_clone_DT, f_clone_DOM)
    w4, p4 = wilcoxon(f_distinct_DT, f_distinct_DOM)

    # compute effect size
    r1, d1 = cohend(f_clone_DT, f_clone_PD)
    r2, d2 = cohend(f_distinct_DT, f_distinct_PD)

    r3, d3 = cohend(f_clone_DT, f_clone_DOM)
    r4, d4 = cohend(f_distinct_DT, f_distinct_DOM)

    print(f'DT(Doc2Vec) vs Visual_PDiff, pos = clone, p: {p1} eff_size: {r1}')
    print(f'DT(Doc2Vec) vs Visual_PDiff, pos = distinct, p: {p2} eff_size: {r2}')
    print(f'DT(Doc2Vec) vs DOM_RTED, pos = clone, p: {p3} eff_size: {r3}')
    print(f'DT(Doc2Vec) vs DOM_RTED, pos = distinct, p: {p4} eff_size: {r4}')

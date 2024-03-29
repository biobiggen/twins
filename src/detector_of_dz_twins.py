import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import glob
import argparse

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.mixture import BayesianGaussianMixture

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
plt.rcParams['font.sans-serif'] = ['Arial']

'''
v1.11
intercept:[ 86.30401172 -18.48227871 -52.08219961]
coef:[[ -2.74810393 -35.98386663]
 [  6.13395705   6.56192771]
 [  0.74538823   9.87496716]]

'''
def remove_outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    #print(np.sum(data<lower_bound))
    #print(np.sum(data>upper_bound))
    return data[(data > lower_bound) & (data < upper_bound)]


def init_database(dst, max_dep=2500):
    data = {}
    f_lst = glob.glob(os.path.join(dst, '*'))
    af_cluster = {}
    for line in f_lst:
        line = line.rstrip()
        ll = os.path.basename(line)
        df = pd.read_csv(line)
        dep = min(np.median(df.REF_DEPTH+df.ALT_DEPTH), max_dep) + 1
        af = df.AF.apply(lambda x: x if x <=0.5 else 1-x)
        bbba = df.loc[df.AF.between(0.01, 0.2), :]
        aaab = df.loc[df.AF.between(0.8, 0.99), :]
        tmp_af = af[af.between(0.01,0.2)]
        tmp_af = list(remove_outliers(tmp_af))
        tmp_af = np.array(tmp_af).reshape(-1, 1)
        res_ff = CBS(tmp_af, min_support=tmp_af.shape[0]/30)
        af_cluster[ll] = res_ff
        tmp_ff = bbba.AF.median()+1-aaab.AF.median()
        raw_std = np.average([bbba.AF.std(), aaab.AF.std()],
                             weights=[bbba.shape[0], aaab.shape[0]])
        cor_std = raw_std * np.sqrt(dep) / np.sqrt(tmp_ff/2*(1-tmp_ff/2))
        hom_tot = np.sum([df.AF < 0.2, df.AF > 0.8])
        if hom_tot > 0:
            x = np.sum(bbba.shape[0]+aaab.shape[0])/hom_tot
        else:
            x = 0
        if ll.split('_')[0] == 'S':
            data[ll] = [ll, x, cor_std, 1]
        else:
            data[ll] = [ll, x, cor_std, 2]
    res = pd.DataFrame(data.values(), columns=['CLI', 'X', 'Y', 'Type'])
    for k in af_cluster:
        if k.startswith('S'):
            continue
        print(f'{k}\n{af_cluster[k]}]')
    return res, af_cluster


def CBS(data, n_clusters=5, min_support=10, gamma=1e2, debug=None):
    if debug:
        print(f'#min_support={min_support}')
    data = np.array(data).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=n_clusters,
                                  init_params='random_from_data', max_iter=1500,
                                  random_state=42,
                                  weight_concentration_prior=gamma).fit(data)
    predict = bgm.predict(data)
    res = []
    weight = {}
    for x in np.unique(predict):
        if np.sum(predict == x) > min_support:
            res.append(np.median(data[predict == x]))
            weight[res[-1]] = np.sum(predict == x)
            xx = np.sort(res)[::-1]
    if debug:
        print(f'# res:{xx}')
    mres = [xx[0]]
    wmres = [weight[xx[0]]]
    for x in xx:
        if x == mres[-1]:
            continue
        if mres[-1] - x < 0.004:
            mres[-1] = np.average([mres[-1], x],
                                  weights=[wmres[-1], weight[x]])
            wmres[-1] += weight[x]
        else:
            mres.append(x)
            wmres.append(weight[x])

    if len(mres) > 3:
        raise Exception(xx)
    elif len(mres) == 3:
        maf1 = mres[2]
        maf2 = mres[1]
        maf3 = mres[0]
    elif len(mres) == 2:   
        maf1 = mres[1]
        maf2 = mres[1]
        maf3 = mres[0]
    else:  # MZ or single #
        maf1 = 0
        maf2 = mres[0]
        maf3 = mres[0]
        #FF_1=(2*bafm_1*bafm_3)/(2*bafm_1*bafm_3+(bafm_1+bafm_2 )*(1-2*bafm_3 ) )                    (18)
        #FF_2=(2*bafm_2*bafm_3)/(2*bafm_2*bafm_3+(bafm_1+bafm_2 )*(1-2*bafm_2 ) )                    (19)

    ff1 = 2*maf1*maf3/(2*maf1*maf3+(maf1+maf2)*(1-2*maf3))
    ff2 = 2*maf2*maf3/(2*maf2*maf3+(maf1+maf2)*(1-2*maf2))
    ret = [ff1, ff2]
    ret = [float('%0.3f' % xx) for xx in ret]
    return(ret)


if __name__ == '__main__':
    dst = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'demo_data')
    print(dst)
    df, _ = init_database(dst=dst)
    df_X = np.array(df.loc[:, ['X', 'Y']])
    df_Y = np.array(df.loc[:, 'Type'])
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y)
    clf = SVC(kernel='rbf', C=1000, probability=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    #  demo png
    plt.scatter(df.X, df.Y, c=df.Type, s=10, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.xlabel('AF variation of fetal SNPs')
    plt.ylabel('Relative abudanse of fetal SNPs')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('twins_svc_demo.png')

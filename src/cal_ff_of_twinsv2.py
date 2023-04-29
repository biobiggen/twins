import os
import sys
import platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture

# fix var = 1 standard normal distribution
# with prior mu conjugate to a normal distribution


def cal_xxxy(afx, afy):
    f1 = 2 * afx/(1-afx-afy-afx*afy)
    f2 = 2 * (1+afx)*afy/(1-afx-afy-afx*afy)
    ff1 = 1/(1/f1+1)
    ff2 = 1/(1/f2+1)
    return ff1, ff2


'''
双胎胎儿百分比转化

假设，胎儿1DNA含量与母亲DNA含量比值为R1,胎儿2DNA含量与母亲DNA含量比值为R2,R1<=R2.
对于正常常染色体，考虑母亲BB
AF1 - BBBABB
AF2 - BBBBBA
AF3 - BBBABA

AF1 = R1/2/(1+R1+R2)
AF2 = R2/2/(1+R1+R2)
AF3 = (R1+R2)/2/(1+R1+R2)

AF1/AF2 = R1/R2
AF1/AF3 = R1/(R1+R2)
AF2/AF3 = R2/(R1+R2)

R1 = 2*AF1/(1-2*AF1-2*AF2) = 2AF1/(1-2*AF3) = 2*AF1(AF3-AF2)/(AF3-AF2-2*AF1*AF3)
R2 = 2*AF2/(1-2*AF1-2*AF2) = 2AF2/(1-2*AF3) = 2*AF2(AF3-AF1)/(AF3-AF1-2*AF2*AF3)
FF1 = R1/(1+R1)
FF2 = R2/(1+R2)

XX|XX/XY|XY
假设，胎儿1DNA含量与母亲DNA含量比值为R1,胎儿2DNA含量与母亲DNA含量比值为R2,#R1<=R2.
对于正常性染色体，不妨设胎儿1为XX,胎儿2为XY，考虑母亲BB
对于X染色体
AFx = R1/2/(1+R1+R2/2) #*
FFx = 2*AFx = R1/(1+R1+R2/2) # error !!! 双胎无2倍 关系?
对于Y染色体
AFy = depY/(depX+depY) = R2/2/(1+R1+R2) #*与单胎不同
FFy = 2*AFy = R2/(1+R1+R2)
AFy1 = depY/DepX

R1*(1-2*AFx) = (2+R2)*AFx = (1+R2/2)*FFx
R2*(1-AFy) = (2+2R1)*AFy

R1 = FFx(1+R2/2)/(1-FFx) 
R2 = FFy(1+R1)/(1-FFy)
## XX|XX
AFx = (R1+R2)/2/(1+R1+R2)
FFx = (R1+R2)/(1+R1+R2)

## XY|XY
AFy = depY/(depX+depY) = (R1+R2)/2/(1+R1+R2)
FFy = (R1+R2)/(1+R1+R2)

Agglomerative Hierarchical Clustering,AHC 合成聚类算法（自下而上）


## 龙凤胎
假设，胎儿1DNA含量与母亲DNA含量比值为R1,胎儿2DNA含量与母亲DNA含量比值为R2,
'''

def CBS(data, afx=0, afy=0, n_clusters=15, min_support=5, gamma=1e1, debug=None):
    # 自上而下
    data = np.array(data).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=5, reg_covar=0,
                                  init_params='random', max_iter=1500,
                                  covariance_type= 'spherical',
                                  random_state=42,
                                  weight_concentration_prior=gamma).fit(data)
    predict = bgm.predict(data)
    res = []
    for x in np.unique(predict):
        if np.sum(predict==x) > min_support:
            res.append(np.median(data[predict == x]))
    """
    ww = []
    tot_af = afx+afy
    if platform.system() == 'Windows':
        debug = True
    weights_sort = [bgm.weights_[x] for x in np.argsort(bgm.weights_)[::-1]]
    if debug:
        print(weights_sort)
    c = 0  #  cycle for best gamma
    # if af2 == 0:
    #   MZ or singleton
    # if af1 = 0:
    #   error
    while(True):
        c += 1
        if c > 1000:
            if debug:
                print(f'#C={c}')
            break
        if debug:
            print('loop start !!!')
            print(f'#gamma0:{gamma}')
            print(bgm.weights_)
        if np.sum(weights_sort[:3]) > 0.99:
            if weights_sort[1] > 0.33:
                break
            else:
                gamma *= 1.01
                continue
        else:
            gamma *= 0.99
        bgm = BayesianGaussianMixture(n_components=n_clusters, reg_covar=0,
                                      init_params='random', max_iter=1500, mean_precision_prior=0.8,
                                      #covariance_type= 'spherical',
                                      random_state=42, weight_concentration_prior=gamma).fit(data)
        predict = bgm.predict(data)
        res = []
        ww = []
        weights_sort = [bgm.weights_[x]
                        for x in np.argsort(bgm.weights_)[::-1]]
        if debug:
            print(weights_sort)
            print(f'#end:gamma1:{gamma}')
            print(bgm.weights_)
        if gamma < 30:
            break
    if debug:
        print(bgm.weights_)
    for x in np.argsort(bgm.weights_)[::-1]:
        if bgm.weights_[x] < 0.1 or np.sum(predict == x) < min_support:
            break
        af = np.median(data[predict == x])
        if debug:
            print('#summary')
            print(np.sum(predict == x))
            print(bgm.weights_[x])
            print(af)
        res.append(af)
        ww.append(np.sum(predict == x))
    """
    xx = np.sort(res)
    if debug:
        print(f'# res:{xx}')
    if len(xx) > 3:
        raise Exception(xx)
        # 聚类错误
    elif len(xx) == 3:
        maf2 = xx[1]
        maf1 = xx[0]
        maf3 = xx[2]
    elif len(xx) == 2:  # ff 接近  # 假设不存在 完美避开
        maf2 = xx[0]
        maf1 = xx[0]
    else:  # MZ or single
        maf2 = xx[0]
        maf1 = xx[0]
    ff1 = maf1/(1-maf2)
    ff2 = maf2/(1-maf1)
    ret = [ff1, ff2]
    ret = [float('%0.3f' % xx) for xx in ret]
    return(ret)


def calculateTwinsFetalFraction(data, ff_threshold=0.025,
                                exclude=['chrY'], tot_ff=None):
    ffx = data.ff['chrX']
    ffy = data.ff_y
    FF3 = []
    if ffx > ff_threshold and ffy > ff_threshold:
        # XXXY
        R1 = ffx*(1-ffy/2)/(1-ffx-ffy+ffx*ffy/2)
        R2 = ffy/(1-ffx-ffy+ffx*ffy/2)
        ff1 = R1/(1+R1)
        ff2 = R2/(1+R2)
        ff1 = float('%0.3f' % ff1)
        ff2 = float('%0.3f' % ff2)
        FF3 = f'FF1:{ff1};FF2:{ff2}'
    elif ffx > ff_threshold and ffy <= ff_threshold:
        # XXXX
        FF3 = ffx
        FF3 = '%0.3f' % FF3
        FF3 = 'X:'+FF3
    elif ffx <= ff_threshold and ffy > ff_threshold:
        FF3 = ffy
        FF3 = '%0.3f' % FF3
        FF3 = 'Y:' + FF3
    else:
        raise Exception('not plasma sample of pregant woman ...')
    af_list = []
    if not hasattr(data, 'qc_twins'):
        data.qc_of_twins()
    if data.qc_twins[1] > 6:
        return ['na', 'na', 'more than 2']
    if not hasattr(data, 'std_af'):
        data.calculate_fetal_fraction()
    for chromosome in data.chrom_index:
        if chromosome in exclude:
            continue
        var_sd = data.std_af[chromosome]
        ff = max(1e-3, data.ff[chromosome])
        var_sd /= np.sqrt(ff*(1-ff))
        var_sd *= np.sqrt(data.umi_median_depth)
        if var_sd > 4:
            continue
        tmp_af_bbba = list(data.df[(data.chrom_index[chromosome]) &
                                   (data.group_tag == 'BBBA')].pA_Ratio)
        tmp_af_aaab = list(data.df[(data.chrom_index[chromosome]) &
                                   (data.group_tag == 'AAAB')].pA_Ratio)
        tmp_af_aaab = [1-x for x in tmp_af_aaab]
        af_list.extend(tmp_af_aaab)
        af_list.extend(tmp_af_bbba)
    af_list = [float('%0.3f' % x) for x in af_list]
    if len(af_list) > 50:
        res = CBS(data=af_list, gamma=data.ref_median_depth*2)
        res.append(FF3)
        return res
    else:
        return ['na', 'na', ' Informative SNP is not enough']


if __name__ == '__main__':
    fin = sys.argv[1]
    cli = os.path.basename(fin)
    cli = '_'.join(cli.split('_',maxsplit=3)[:3])
    df = pd.read_table(fin)
    g = df.groupby('#Chr')
    res = []
    autosomes_index = ~df['#Chr'].isin(['chrX', 'chrY'])
    rd_autosomes = df.DepRegion[autosomes_index].median()
    ref_index = ~g.size().index.isin(['chr13', 'chr18', 'chr21', 'chr22',
                                      'chrX', 'chrY'])
    x_df = df.loc[df['#Chr'] == 'chrX', ]
    rd_xchr = x_df.DepRegion.median()
    x_alt = x_df.loc[x_df.pA_Ratio.between(0.015, 0.2), 'Alt_Dep']
    x_ref = x_df.loc[x_df.pA_Ratio.between(0.8, 0.985), 'Ref_Dep']
    rd_x = []
    rd_x.extend(list(x_alt.values))
    rd_x.extend(list(x_ref.values))
    rd_x = np.median(rd_x)
    rd_y = df.loc[df['#Chr'] == 'chrY', 'DepRegion'].median()
    af_x = rd_x / rd_autosomes
    af_y = rd_y/rd_autosomes/0.84
    if af_x < 0.01:
        af_x = 0
    if af_y < 0.01:
        af_y = 0
    auto_pa = df.loc[autosomes_index, 'pA_Ratio']
    auto_pa = [x if x < 0.5 else 1-x for x in auto_pa]
    auto_pa = [x for x in auto_pa if  x > 0.01]
    auto_pa = [x for x in auto_pa if  x < 0.25]
    if len(auto_pa) > 50:
        af_auto = CBS(data=auto_pa, afx=af_x, afy=af_y)
    else:
        af_auto= [None,None]
    res = [cli,rd_autosomes,rd_xchr,rd_y,rd_x,af_x,af_y]
    res.extend(af_auto)
    print('\t'.join(map(str,res))+'\n')

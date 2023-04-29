import os
import sys
import argparse
import configparser
os.environ['QT_QPA_PLATFORM']='offscreen'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss


"""plot 4 twins_jmd """
'''
Demo data 
+---------------------------------+
| SNP_ID | Depth |  AF  | QC_Stat |
|--------+-------+------+---------|
|  1     | 2000  | 0.5  | PASS    |
|--------+-------+------+---------|
|  2     | 400   | 0.15 | FAIL    |
|--------+-------+------+---------|
|  3     | 1500  | 0.86 | PASS    |
+--------+-------+------+---------+
'''

def get_args():
    parser      =   argparse.ArgumentParser()
    parser.add_argument('--type',choices=['singleton','twins'],help='sample type')
    parser.add_argument('--fetalfraction1','-f1',type=float,help='FF')
    parser.add_argument('--fetalfraction2','-f2',type=float,help='FF')
    parser.add_argument('--png','-p',default='test.png',help='output png')
    parser.add_argument('--dep',type=int,default=2000,help='the mean depth ')
    parser.add_argument('--out',default='test.xls',help='output data')
    parser.add_argument('--seed',default=42,type=int,help='SEEDS')
    parser.add_argument('--batch',action='store_true',help='batch mode')
    return parser.parse_args()

def get_genotype(maf=(0.5,0.5),n=100,cp=2):
    ab_lst = ['A','B']
    return np.random.choice(ab_lst,replace=True,size=(n,cp),p=maf).T


class depth_simulate():
    '''
    depth simulator
    '''
    def __init__(self,m_depth=1000,n=1000,var_depth=50,methods='possion'):
        if methods == 'norm':
            self.depth=ss.norm.rvs(loc=m_depth,scale=var_depth,size=n)
        elif methods=='possion':
            self.depth = np.random.poisson(m_depth,n)
    def get_depth(self):
        return self.depth


def get_child_genotype(gm,gf,cpm=1,cpp=1,fgm=None,fgp=None):
    assert gm.shape[1] == gf.shape[1],'SNPs no. not equal! '
    if fgm is None:
        fgm     =   np.random.choice(gm.shape[0],(cpm,2))
    if fgp is None:
        fgp     =   np.random.choice(gf.shape[0],(cpp,2))
    res     =   [''] * gm.shape[1]
    bp      =   gm.shape[1] >> 1
    comb_p = np.random.random() > 0.5
    comb_m = np.random.random() > 0.5
    for x in fgm:
        for y in np.arange(gm.shape[1]):
            if comb_m:
                if y < bp:
                    res[y] += gm[x[0]][y]
                else:
                    res[y] += gm[x[1]][y]
            else:
                res[y] += gm[x[0]][y]
    for x in fgp:
        for y in np.arange(gm.shape[1]):
            if comb_p:
                if y < bp:
                    res[y] += gf[x[0]][y]
                else:
                    res[y] += gf[x[1]][y]
            else:
                res[y] += gf[x[0]][y]
    return res

def get_af(child_n=1,ff=[0.1],n=100,n_chrom=20,fgm=None,fgp=None):
    assert len(ff)  ==   child_n
    gm      =   get_genotype(n=n)
    gf      =   get_genotype(n=n)
    mf      =   1 - np.sum(ff)
    tot_a   =   np.sum(gm=='A',axis=0,dtype='float64') * mf
    tot_a   =   np.tile(tot_a,n_chrom)
    child_gt = [[] for x in  range(child_n)]
    for cc in np.arange(n_chrom):
        for x in np.arange(child_n):
            child_gt[x].extend(get_child_genotype(gm=gm,gf=gf,fgm=fgm,fgp=fgp))
    for x in np.arange(child_n):
        tmp_fa = np.array([item.count('A') for item in child_gt[x]]) * ff[x]
        tot_a = tot_a + tmp_fa
    afs     =   tot_a/2
    return afs
    

def plot_png(out='test.png',w=18,h=9,f1=.09,f2=0.1,s=3):
    fig,axes        =   plt.subplots(2,2,figsize=(w,h))
    # s1
    n=800
    fgm1 =  np.array([[0,0],[0,0]])
    fgp1 =  np.array([[0,0],[0,0]])
    af1  =  get_af(ff=[f1,f2],child_n=2,n=n,fgm=fgm1,fgp=fgp1)
    af11 =  get_af(ff=[f1],child_n=1,n=n,fgm=fgm1,fgp=fgp1)
    fgm2 =  np.array([[0,0],[1,1]])
    fgp2 =  np.array([[0,0],[0,0]])
    af2  =  get_af(ff=[f1,f2],child_n=2,n=n,fgm=fgm2,fgp=fgp2)
    fgm3 =  np.array([[0,0],[0,0]])
    fgp3 =  np.array([[0,0],[1,1]])
    af3  =  get_af(ff=[f1,f2],child_n=2,n=n,fgm=fgm3,fgp=fgp3)
    fgm4 =  np.array([[0,0],[1,1]])
    fgp4 =  np.array([[0,0],[1,1]])
    af4  =  get_af(ff=[f1,f2],child_n=2,n=n,fgm=fgm4,fgp=fgp4)
    axes[0][0].scatter(x=np.arange(n),y=af1,s=s)
    axes[0][0].set_title('a')
    axes[0][1].scatter(x=np.arange(n),y=af2,s=s)
    axes[0][1].set_title('b')
    axes[1][0].scatter(x=np.arange(n),y=af3,s=s)
    axes[1][0].set_title('c')
    axes[1][1].scatter(x=np.arange(n),y=af4,s=s)
    axes[1][1].set_title('d')
    for x in axes:
        for y in x:
            y.set_xticks([])
            y.set_ylabel('MAF')
    plt.savefig(out)
    plt.cla()
    fig2,axes2        =   plt.subplots(1,1,figsize=(w/2,h/2))
    axes2.scatter(x=np.arange(n),y=af11,s=s)
    axes2.set_xticks([])
    plt.ylabel('MAF')
    plt.savefig(f'normal_{out}')

def main():
    args        =   get_args()
    max_ff = 0.2
    np.random.seed(args.seed)
    m_depth = args.dep  
    if args.batch:##BATCH MPDE ##
        n_twins = 100
        n_singleton = 100
        n_snps = 2000
        n_chrom = 20
        n1 = int(n_snps/n_chrom)
        n2 = n1 * n_chrom
        ff0 = np.mod(np.random.random(size=n_singleton),max_ff-0.04)+0.04
        ff1 = np.mod(np.random.random(size=n_twins),max_ff-0.04)+0.04
        ff2 = np.mod(np.random.random(size=n_twins),max_ff-0.04)+0.04
        for x in np.arange(n_singleton):
            raw_af = get_af(ff=[ff0[x]],n=n1)
            depth = np.random.poisson(m_depth,n2)
            sample_id = f'S_{x}_ff_{ff0[x]:.2f}'
            fhout = open(f'{sample_id}_demo.csv','w')
            fhout.write(','.join(['SNP_ID','REF_DEPTH','ALT_DEPTH','AF'])+'\n')
            snp_id = 0
            for i,j in zip(depth,raw_af):
                snp_id += 1
                M = i * 4
                ngood = M * j
                nbad  = M - ngood
                nalt = np.random.hypergeometric(ngood=ngood,nbad=nbad,nsample=i)
                nref = i - nalt
                new_af = float(nalt/i)
                new_af = '%0.4f' % new_af
                fhout.write(','.join(map(str,[f'snp_{snp_id}',nref,nalt,new_af]))+'\n')
            fhout.close()
        for x in np.arange(n_twins):
            raw_af = get_af(child_n=2,ff=[ff1[x],ff2[x]],n=n1)
            depth = np.random.poisson(m_depth,n2)
            sample_id = f'Twins_{x}_ff1_{ff1[x]:.2f}_ff2_{ff2[x]:.2f}'
            fhout = open(f'{sample_id}_demo.csv','w')
            fhout.write(','.join(['SNP_ID','REF_DEPTH','ALT_DEPTH','AF'])+'\n')
            snp_id = 0
            for i,j in zip(depth,raw_af):
                snp_id += 1
                M = i * 4
                ngood = M * j
                nbad  = M - ngood
                nalt = np.random.hypergeometric(ngood=ngood,nbad=nbad,nsample=i)
                nref = i - nalt
                new_af = float(nalt/i)
                new_af = '%0.4f' % new_af
                fhout.write(','.join(map(str,[f'snp_{snp_id}',nref,nalt,new_af]))+'\n')
            fhout.close()
    else:

        ff1          =   args.fetalfraction1
        ff2          =   args.fetalfraction2
        if ff1 is None:
            ff1 = max(0.04,np.mod(np.random.random(),max_ff))
        if ff2 is None:
            ff2 = max(0.04,np.mod(np.random.random(),max_ff))
        if args.type == 'twins':
            fetus_n     =   2
            ff = [ff1,ff2]
        else:
            fetus_n     =   1
            ff = [ff1]
        n_snps = 1000
        n_chrom = 20
        n1 = int(n_snps/n_chrom)
        n2 = n1 * n_chrom
        depth = np.random.poisson(args.dep,n2)
        raw_af = get_af(child_n=fetus_n,ff=ff,n=n1)
        fhout = open(f'{args.out}','w')
        for i,j in zip(depth,raw_af):
            snp_id += 1
            M = i * 4
            ngood = M * j
            nbad  = M - ngood
            nalt = np.random.hypergeometric(ngood=ngood,nbad=nbad,nsample=i)
            nref = i - nalt
            new_af = float(nalt/i)
            new_af = '%0.4f' % new_af
            fhout.write(','.join([f'snp_{snp_id}',nref,nalt,new_af])+'\n')
        fhout.close()



if __name__ ==  '__main__':
    main()


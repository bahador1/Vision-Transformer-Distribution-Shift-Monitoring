"""Run the digit experiment, or supply a CIFAR embedding archive."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, cdist
from scipy.special import xlogy
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def run(seed, archive, trials, output):
    rng = np.random.default_rng(seed)
    if archive:
        a = np.load(archive)
        base, y = a['clean'], a['labels']
        conditions = {k: a[k] for k in a.files if k.startswith('shift_')}
        training_condition = 'shift_motion_blur_3'
    else:
        images, y = load_digits(return_X_y=True)
        base = images / 16
        conditions = {f'blur_{s}': gaussian_filter(base.reshape(-1,8,8),(0,s,s)).reshape(-1,64) for s in (.35,.65,1.)}
        conditions['brightness'] = np.clip(base + .15,0,1)
        training_condition = 'blur_0.65'
    clean = normalize(base)
    conditions = {k:normalize(x) for k,x in conditions.items()}
    fit, left, right, ref, target = np.array_split(rng.permutation(len(y)),5)
    probe = LogisticRegression(C=10,max_iter=1000).fit(clean[fit],y[fit])
    left,right = left[:min(len(left),len(right))],right[:min(len(left),len(right))]
    u,v = clean[left],conditions[training_condition][right]
    width = np.median(pdist(clean[fit[:1000]]))
    def kernel(x):
        return np.exp(-cdist(x,x,'sqeuclidean')/(2*width**2))
    direction = v.mean(0)-u.mean(0)
    direction /= max(np.linalg.norm(direction),1e-12)
    pool = np.vstack([u,v])
    coord = pool @ direction
    scale = max(np.std(coord),1e-8)
    def adapted(x, weight):
        d = np.subtract.outer(x@direction,x@direction)/scale
        return kernel(x)*(.05+.95*np.exp(-weight*d*d/2))
    scores = []
    for weight in [0,.125,.5,2,8]:
        k = adapted(pool,weight)
        n = len(u)
        h = k[:n,:n]+k[n:,n:]-k[:n,n:]-k[n:,:n]
        np.fill_diagonal(h,0)
        rows = h.sum(1)/(n-1)
        scores.append(rows.mean()/np.sqrt(4*rows.var()+1e-8))
    weight = [0,.125,.5,2,8][np.argmax(scores)]
    with open(output,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed','weight','n','condition','trial','method','p','accuracy','clean_accuracy'])
        for n in ([32,64,128] if archive else [32]):
            for trial in range(trials):
                r,t = rng.choice(ref,n,False),rng.choice(target,n,False)
                signs = np.r_[np.ones(n),-np.ones(n)]/n
                s = np.array([signs]+[rng.permutation(signs) for _ in range(199)])
                for name,data in {'clean':clean,**conditions}.items():
                    x = np.vstack([clean[r],data[t]])
                    p = probe.predict_proba(x)
                    entropy = -xlogy(p,p).sum(1)
                    stats = {'fixed':np.sum((s@kernel(x))*s,1),'learned':np.sum((s@adapted(x,weight))*s,1),'centroid':np.linalg.norm(s@x,axis=1),'entropy':abs(s@entropy)}
                    for method,values in stats.items():
                        w.writerow([seed,weight,n,name,trial,method,(1+sum(values[1:]>=values[0]-1e-12))/200,probe.score(data[t],y[t]),probe.score(clean[t],y[t])])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trials',type=int,default=100); p.add_argument('--output',default='results.csv')
    a=p.parse_args(); run(a.seed,a.archive,a.trials,a.output)

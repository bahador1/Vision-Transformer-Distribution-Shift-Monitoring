"""Window-level two-sample monitoring. See README.md for both entry points."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def permutation_tests(ref, target, ent_ref, ent_target, bandwidth, permutations, rng):
    """Balanced label shuffles. Row 0 is the observed allocation."""
    n = len(ref)
    z = np.concatenate((ref, target)).astype(np.float64)
    entropy = np.r_[ent_ref, ent_target]
    signs = np.tile(np.r_[np.ones(n), -np.ones(n)], (1 + permutations, 1))
    for row in signs[1:]:
        rng.shuffle(row)
    weights = signs / n
    start = perf_counter()
    kernel = np.exp(-squareform(pdist(z, 'sqeuclidean')) / (2 * bandwidth**2))
    mmd = np.einsum('ij,ij->i', weights @ kernel, weights)
    t_mmd = perf_counter() - start
    start = perf_counter()
    centroid = np.linalg.norm(weights @ z, axis=1)
    t_centroid = perf_counter() - start
    start = perf_counter()
    ent = abs(weights @ entropy)
    t_ent = perf_counter() - start
    for name, values, elapsed in [('MMD', mmd, t_mmd), ('Centroid', centroid, t_centroid),
                                   ('Entropy', ent, t_ent)]:
        ties = np.isclose(values[1:], values[0], rtol=1e-12, atol=1e-14)
        p = (1 + np.count_nonzero((values[1:] > values[0]) | ties)) / len(values)
        yield name, float(values[0]), p, 1000 * elapsed


def synthetic(a, meta):
    rng = np.random.default_rng(345)
    centers = rng.normal(size=(10, 20))
    def sample(n):
        y = rng.integers(10, size=n)
        return centers[y] + rng.normal(size=(n, 20)), y
    train, y_train = sample(a.fit)
    clean, y = sample(2 * a.pool)
    def conditions():
        for severity in a.severities:
            shifted = clean[a.pool:].copy()
            shifted[:, 0] += severity
            yield 'synthetic_offset', severity, unit(shifted)
    return unit(train), y_train, unit(clean[:a.pool]), unit(clean[a.pool:]), y[a.pool:], conditions()


def real(a, meta):
    import torch
    from torchvision.datasets import CIFAR10
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = AutoModel.from_pretrained(a.model, revision=a.revision).to(device).eval()
    revision = getattr(model.config, '_commit_hash', None) or a.revision
    processor = AutoImageProcessor.from_pretrained(a.model, revision=revision, use_fast=False)
    meta.update(device=str(device), model_revision=revision, processor=processor.to_dict(),
                torch=version('torch'), torchvision=version('torchvision'), transformers=version('transformers'))
    train, test = [CIFAR10(a.data, train=flag, download=True) for flag in (True, False)]
    rng = np.random.default_rng(a.seed)
    fit_ids = rng.permutation(len(train))[:a.fit]
    ids = rng.permutation(len(test))[:2*a.pool]
    ref_ids, target_ids = np.split(ids, 2)
    np.savez(a.out/'image_ids.npz', fit=fit_ids, reference=ref_ids, target=target_ids)
    def encode(images, indexes):
        begin = perf_counter()
        pieces = []
        with torch.inference_mode():
            for i in range(0, len(indexes), a.encode_batch):
                batch = [Image.fromarray(images[j]) for j in indexes[i:i+a.encode_batch]]
                inputs = processor(images=batch, return_tensors='pt')
                tokens = model(pixel_values=inputs.pixel_values.to(device)).last_hidden_state
                pieces.append(tokens[:, 0].cpu().float().numpy())
        meta['encode_seconds'] = meta.get('encode_seconds', 0) + perf_counter() - begin
        return unit(np.concatenate(pieces))
    y = np.asarray(test.targets)
    labels = np.load(a.corruption_dir/'labels.npy').reshape(-1)
    if len(labels) not in (10000, 50000) or not np.all(labels.reshape(-1,10000) == y):
        raise ValueError('CIFAR-10-C labels must match torchvision test order in every severity block.')
    def conditions():
        for name in a.corruptions:
            images = np.load(a.corruption_dir/(name+'.npy'), mmap_mode='r')
            if images.shape != (50000,32,32,3) or images.dtype != np.uint8:
                raise ValueError(f'Unexpected CIFAR-10-C array for {name}: {images.shape}, {images.dtype}')
            for severity in a.severities:
                print(f'Encoding {name} severity {severity}', flush=True)
                yield name, severity, encode(images, target_ids + 10000*(severity-1))
    return (encode(train.data, fit_ids), np.asarray(train.targets)[fit_ids], encode(test.data, ref_ids),
            encode(test.data, target_ids), y[target_ids], conditions())


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def experiment(a):
    from itertools import chain
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit('Select an empty output directory to keep runs separate.')
    meta = dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                numpy=version('numpy'), scipy=version('scipy'), sklearn=version('scikit-learn'))
    train, y_train, ref, clean, y, conditions = (synthetic if a.mode=='demo' else real)(a, meta)
    bandwidth = float(np.median(pdist(train[:min(len(train),1000)])))
    if bandwidth <= 0:
        raise ValueError('Zero bandwidth: inspect source embeddings.')
    probe = LogisticRegression(C=10, max_iter=1000).fit(train, y_train)
    def predictions(z):
        probs = probe.predict_proba(z)
        return -(probs*np.log(probs.clip(1e-15))).sum(1), probe.classes_[probs.argmax(1)]
    ent_ref, _ = predictions(ref)
    ent_clean, pred_clean = predictions(clean)
    rows, summary = [], []
    for name, severity, target in chain([('clean',0,clean)], conditions):
        ent_target, pred_target = predictions(target)
        for n in a.windows:
            # Coupled trials allow paired comparisons across conditions and methods.
            rng = np.random.default_rng([a.seed, n])
            group = []
            for trial in range(a.trials):
                ri = rng.choice(len(ref), n, replace=False)
                ti = rng.choice(len(target), n, replace=False)
                results = permutation_tests(ref[ri],target[ti],ent_ref[ri],ent_target[ti],
                                            bandwidth,a.permutations,rng)
                for method, score, p, ms in results:
                    group.append(dict(condition=name,severity=severity,window=n,trial=trial,
                        method=method,score=score,p=p,alarm=int(p<=a.alpha),score_ms=ms,
                        accuracy=np.mean(pred_target[ti]==y[ti]),
                        clean_accuracy=np.mean(pred_clean[ti]==y[ti])))
            rows.extend(group)
            for method in ('MMD','Centroid','Entropy'):
                sub = [r for r in group if r['method']==method]
                summary.append(dict(condition=name,severity=severity,window=n,method=method,
                    alarms=sum(r['alarm'] for r in sub),trials=a.trials,
                    alarm_rate=np.mean([r['alarm'] for r in sub]),
                    accuracy=np.mean(pred_target==y),clean_accuracy=np.mean(pred_clean==y),
                    median_score_ms=np.median([r['score_ms'] for r in sub])))
    write_rows(a.out/'trials.csv',rows)
    write_rows(a.out/'summary.csv',summary)
    meta.update(bandwidth=bandwidth,feature_dim=train.shape[1])
    (a.out/'run.json').write_text(json.dumps(meta,indent=2,default=str))
    for name in sorted({r['condition'] for r in summary}-{'clean'}):
        fig, axes = plt.subplots(1,len(a.windows),figsize=(4*len(a.windows),3.5),squeeze=False,sharey=True)
        for ax,n in zip(axes[0],a.windows):
            for method in ('MMD','Centroid','Entropy'):
                pts=sorted((r['severity'],r['alarm_rate']) for r in summary if
                    r['window']==n and r['method']==method and r['condition'] in ('clean',name))
                ax.plot(*zip(*pts),marker='o',label=method)
            ax.axhline(a.alpha,color='gray',ls=':',label='Nominal alpha')
            ax.set(xlabel='Severity (0 = clean)',ylim=(-.04,1.04),title=f'{name}, n={n}')
        axes[0,0].set_ylabel('Fraction of windows flagged')
        axes[0,-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out/(name+'.png'),dpi=160)
        plt.close(fig)
    print(f'Wrote {a.out}/summary.csv, trials.csv, run.json and figures.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['demo','cifar'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',default='./data')
    p.add_argument('--corruption-dir',type=Path,default=Path('data/CIFAR-10-C'))
    p.add_argument('--corruptions',nargs='+',default=['gaussian_noise','motion_blur','brightness'])
    p.add_argument('--severities',type=int,nargs='+',default=[1,3,5])
    p.add_argument('--windows',type=int,nargs='+',default=[32,64])
    p.add_argument('--fit',type=int,default=3000)
    p.add_argument('--pool',type=int,default=1000)
    p.add_argument('--trials',type=int,default=50)
    p.add_argument('--permutations',type=int,default=199)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',help='cpu or cuda (automatic by default)')
    p.add_argument('--encode-batch',type=int,default=32)
    p.add_argument('--model',default='facebook/dinov2-small')
    p.add_argument('--revision',default='main')
    a=p.parse_args()
    if not (20<=a.fit<=50000 and 2<=a.pool<=5000 and all(2<=n<=a.pool for n in a.windows)
            and all(1<=s<=5 for s in a.severities) and a.trials>=1 and a.encode_batch>0
            and a.seed>=0 and 0<a.alpha<1 and a.permutations>=1 and 1/(1+a.permutations)<=a.alpha):
        p.error('Check fit, pool, window, severity, alpha and permutation count (minimum p must reach alpha).')
    if a.mode=='cifar':
        for file in ['labels',*a.corruptions]:
            if not (a.corruption_dir/(file+'.npy')).is_file():
                p.error(f'Missing {file}.npy under {a.corruption_dir}. See README.md.')
    with threadpool_limits(limits=1):
        experiment(a)

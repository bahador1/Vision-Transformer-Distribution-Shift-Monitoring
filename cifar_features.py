"""Encode matching CIFAR-10 test identities and their corrupted views."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from transformers import AutoImageProcessor, AutoModel

p=argparse.ArgumentParser()
p.add_argument('data',type=Path,help='parent of CIFAR-10-C directory')
p.add_argument('--count',type=int,default=2000)
p.add_argument('--revision',default='main')
p.add_argument('--output',default='features.npz')
a=p.parse_args()
if not 650<=a.count<=10000: p.error('count must be 650..10000 for 128-image windows')
source=CIFAR10(str(a.data),train=False,download=True)
ids=np.random.default_rng(2026).choice(10000,a.count,False)
directory=a.data/'CIFAR-10-C'
labels=np.load(directory/'labels.npy').reshape(-1,10000)
assert np.all(labels==np.asarray(source.targets)), 'CIFAR test order mismatch'
name='facebook/dinov2-small'
processor=AutoImageProcessor.from_pretrained(name,revision=a.revision)
device='cuda' if torch.cuda.is_available() else 'cpu'
model=AutoModel.from_pretrained(name,revision=a.revision).to(device).eval()
@torch.inference_mode()
def encode(images):
    parts=[]
    for batch in np.array_split(images,int(np.ceil(len(images)/32))):
        inputs=processor(images=[Image.fromarray(x) for x in batch],return_tensors='pt').to(device)
        parts.append(model(**inputs).last_hidden_state[:,0].cpu().numpy())
    return np.concatenate(parts)
out=dict(clean=encode(source.data[ids]),labels=np.asarray(source.targets)[ids],ids=ids,revision=str(model.config._commit_hash))
for family in ('motion_blur','brightness','gaussian_noise'):
    images=np.load(directory/(family+'.npy'),mmap_mode='r')
    for s in range(1,6):
        out[f'shift_{family}_{s}']=encode(images[10000*(s-1)+ids])
np.savez(a.output,**out)

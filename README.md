# Visual shift monitoring
Install numpy, scipy and scikit-learn. From this directory run:

    for seed in 42 43 44; do OPENBLAS_NUM_THREADS=1 python experiment.py --seed "$seed" --output "seed$seed.csv"; done

Each CSV contains one record per seed, condition, trial and method. Count p <= 0.05 for each condition and method. For accuracy average the accuracy column for any single method, since the same probe is used for all four.

## CIFAR benchmark
Install torch, torchvision, transformers and Pillow. Extract the official CIFAR-10-C archive under data/CIFAR-10-C. Clean CIFAR-10 and facebook/dinov2-small are downloaded on first use. Set --revision to a checkpoint commit for repeatable feature extraction.

    python cifar_features.py data --output features.npz
    OPENBLAS_NUM_THREADS=1 python experiment.py --archive features.npz --seed 42 --trials 200 --output cifar42.csv

The seed CSVs shipped here contain digits results. The CIFAR path requires a separate run.

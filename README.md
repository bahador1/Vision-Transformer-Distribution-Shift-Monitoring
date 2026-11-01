# Vision-Transformer-Distribution-Shift-Monitoring

Implemented DINOv2-based distribution shift detection with deep kernel MMD permutation tests for a loading dock monitoring system; developed a reproducible CIFAR-10/CIFAR-10-C demonstration

## Research Question

At the same false-alarm rate, does MMD on frozen DINOv2 features detect image corruptions more reliably than simpler methods, and do its alarms correspond to a loss in classification accuracy?

## Approach

This project is a **comparative evaluation of shift detectors**, built around the MMD test:

- **Detectors compared:** MMD on frozen DINOv2 features vs. simpler baselines, such as the distance between the mean feature vectors of the two groups.
- **Fair comparison:** every detector is calibrated to the same false-alarm rate (5%) and evaluated on the same batches of clean and corrupted (e.g. blurred) images.
- **Detectable vs. harmful shift:** classification accuracy is measured separately, to tell whether a detected change actually hurts the model.

## What the Results Would Show

- If MMD catches mild corruptions more often, that shows when its extra complexity pays off.
- If both detectors perform equally, that shows when the simpler method is enough.

## Status

The full DINOv2 / CIFAR-10-C experiment has not been run yet, so which detector performs better is still an open question.

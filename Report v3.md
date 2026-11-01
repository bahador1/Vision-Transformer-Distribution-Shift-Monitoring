# Detecting Distribution Shift in Loading Dock Camera Streams

_A controlled evaluation with DINOv2, CIFAR-10, and CIFAR-10-C_

## Abstract

A loading dock camera monitor compares incoming images with a reference recorded during normal operation. This study evaluates its statistical decision on public data. Four detectors compare clean CIFAR-10 images with images from the fog corruption in CIFAR-10-C using frozen DINOv2 representations. The evaluation separates the probability of detecting a shift from the classifier’s loss of accuracy.

At fog severity 2, a learned maximum mean discrepancy $MMD$ test flagged 187 of 300 windows, compared with 68 for fixed Gaussian MMD. On clean images, the same detectors raised 3 and 10 false alarms, respectively. The advantage depended on severity. At the mildest fog level, fixed MMD flagged more windows. At the two highest levels, both tests flagged every window, although classification accuracy continued to fall. These results identify a range in which learning the kernel improved sensitivity and a range in which alarm frequency ceased to distinguish the consequences of a shift.

## 1. Application and research questions

Changes in visibility can affect a loading dock camera before labels become available to evaluate its classifier. The monitor therefore makes one decision for a window of frames. It compares current images with reference images collected under accepted operating conditions. An alarm prompts an operator to inspect the images and their predictions.

The experiment addresses three questions:

1. Does a kernel learned from historical examples of reduced visibility detect fog more frequently than a fixed Gaussian kernel on previously unseen images?
    
2. How do the two kernel tests compare with simpler detectors based on the mean image representation or the classifier’s average predictive entropy?
    
3. As fog becomes more severe, how closely does the frequency of alarms track the loss of classification accuracy?
    

CIFAR-10 and CIFAR-10-C provide matched clean and corrupted views of the same scenes. This permits a controlled comparison of detectors and a separate measurement of classification harm. The performance estimates below apply to the public benchmark. Industrial camera performance requires field measurements.

## 2. Related work and statistical method

Gretton et al. [1] developed MMD for testing whether two samples come from the same distribution. Rabanser et al. [2] studied dataset shift detection after mapping observations into a feature space. The present study follows that approach. A frozen DINOv2 ViT-S/14 encoder [3] produces a 384-dimensional class token for each image. Images are resized to 256 pixels, center-cropped to 224 pixels, normalized with the checkpoint’s channel statistics, and processed with its quantized ONNX export. Each resulting feature vector is normalized to unit length.

### 2.1 Fixed Gaussian MMD

Let $u_1,\ldots,u_n$ denote reference features and $v_1,\ldots,v_n$ denote target features. The null hypothesis is that the two sets were drawn from the same feature distribution:

$$
H_0^R=P_Z^T,  
\qquad  
H_1^R\ne P_Z^T.
$$

The fixed Gaussian kernel measures similarity between two features as

$$
k_0(u,v)=  
\exp!\left(-\frac{|u-v|_2^2}{2\sigma^2}\right),
$$

where $\sigma$ is the median pairwise distance among the clean fitting examples. The empirical squared MMD compares average similarity within each group with average similarity across groups:

$$\frac{2}{n^2}\sum_{i,j}k(u_i,v_j)$$.  


The implementation computes $D=w^\top K w$, where $K$ is the pooled kernel matrix and $w$ assigns weight $+1/n$ to each reference image and $-1/n$ to each target image.

### 2.2 Learning a kernel for reduced visibility

Liu et al. [5] learned deep kernels to increase the power of two-sample tests. We implement a restricted version of their idea. A fixed Gaussian kernel treats equal feature distances equally in every direction. Historical examples of fog provide a way to estimate whether particular directions deserve greater weight.

On separate development images, let $d$ be the unit vector pointing from the mean clean feature to the mean severity-3 fog feature. Let $s$ be the pooled standard deviation of the development features projected onto $d$. The candidate kernel is

$$k_0(u,v)  
\left[  
0.05+  
0.95\exp!\left(  
-\frac{a[d^\top(u-v)]^2}{2s^2}  
\right)  
\right]$$.  
]

When $a=0$, the expression reduces to the fixed Gaussian kernel. Positive values reduce the similarity assigned to image pairs that differ along the learned direction. We select $a$ from ${0,0.125,0.5,2,8}$.

For each candidate, define

$$k(u_i,u_j)+k(v_i,v_j)-k(u_i,v_j)-k(v_i,u_j),  
\qquad i\ne j$$,  
]

and let $r_i$ be the mean of the off-diagonal entries in row $i$. The selected weight maximizes

$$
\frac{\bar r}{\sqrt{4\operatorname{Var}(r_i)+10^{-8}}}.
$$

This criterion favors a discrepancy that is large relative to its variability. The direction and weight are frozen before evaluating the held-out windows.

### 2.3 Comparators and calibration

The **centroid detector** measures the Euclidean distance between the mean reference and target features. The **entropy detector** measures the absolute difference in mean predictive entropy from a logistic classifier trained on clean features.

All four detectors receive the same images and the same 199 balanced random permutations. For the observed score $D_{\mathrm{obs}}$ and shuffled scores $D_b$, the permutation p-value is

$$
p=  
\frac{1+\sum_{b=1}^{199}\mathbf{1}{D_b\ge D_{\mathrm{obs}}}}{200}.
$$

A detector raises an alarm when $p\le0.05$. For a clean target window, an alarm counts as a false positive. For a fogged target window, it counts as a detection. Thus, **5% is the nominal significance level per window**. The actual fraction of clean windows flagged is measured separately for each detector.

## 3. Experimental design

The experiment uses the first **320 CIFAR-10 test identities** and their corresponding CIFAR-10-C fog images at severities 1 through 5 [4]. Each severity contains the same target identities. Differences across severity levels therefore reflect changes to the images while holding their class composition fixed. At severity 1, a nearest-image correlation check recovered the corresponding clean identity for all 320 images.

  

  

  

_Figure 1. Each row follows one CIFAR-10 test image from its clean version through the five official fog severities._

For each of three random seeds (42, 43, and 44), the 320 identities were divided into five disjoint pools of 64:

|Pool|Role|
|---|---|
|Clean fitting pool|Train the logistic classifier and set the fixed Gaussian bandwidth.|
|Clean development pool|Estimate the clean side of the learned kernel.|
|Fog development pool|Estimate the shifted side using severity-3 fog.|
|Reference pool|Supply clean images for evaluation windows.|
|Target pool|Supply held-out identities evaluated under clean conditions and at each fog severity.|

Each trial draws 32 reference images and 32 target images. The four detectors decide whether to raise an alarm without access to target class labels. The labels are then used to measure accuracy on the clean and fogged versions of the same target images.

Each split contributes 100 trials per condition, giving **300 window comparisons per condition**. Images can appear in several trials. Accuracy is the mean target-window accuracy, and accuracy loss is the paired difference from the clean condition in percentage points.

## 4. Results

**Table 1.** Alarm counts out of 300 windows per condition. The clean row reports false alarms. Accuracy and accuracy loss refer to the classifier evaluated on the same target windows.

|Condition|Fixed MMD|Learned MMD|Centroid|Entropy|Accuracy|Accuracy loss|
|---|---|---|---|---|---|---|
|Clean|10|3|11|6|83.08%|0.00 pp|
|Fog 1|13|7|14|9|81.64%|1.45 pp|
|Fog 2|68|187|74|23|76.54%|6.54 pp|
|Fog 3|289|300|290|77|72.48%|10.60 pp|
|Fog 4|300|300|300|227|61.14%|21.95 pp|
|Fog 5|300|300|300|300|43.38%|39.71 pp|

**The largest separation between the kernel tests occurred at severity 2.** Learned MMD detected 62.3% of windows, compared with 22.7% for fixed MMD, 24.7% for the centroid detector, and 7.7% for entropy. On clean windows, learned MMD raised fewer false alarms than fixed MMD (1.0% versus 3.3%). Both tests used the same nominal 5% threshold. The selected kernel weights were 0.5, 2, and 0.5 across the three splits, so every split assigned additional weight to the fog direction.

At severity 1, all detectors had low detection rates. Learned MMD flagged only 7 windows, compared with 13 for fixed MMD. At severity 3, learned MMD flagged all 300 windows. Fixed MMD and the centroid detector also detected most windows at this level, while entropy flagged only 77. The representation had therefore changed enough for the feature-based tests to detect it in many windows where mean predictive uncertainty gave little evidence of a shift.

**Alarm frequency and classification harm diverged at high severity.** Both MMD detectors flagged all 300 windows at severities 4 and 5. Nevertheless, classification accuracy fell from 61.14% to 43.38% between those levels. Once the alarm rate reached 100%, it could no longer distinguish a 21.95-point accuracy loss from a 39.71-point loss. For the loading dock application, this distinction matters: an alarm can justify inspection, but its occurrence alone cannot quantify the operational impact.

## 5. Scope and limitations

This experiment measures one corruption family, one feature extractor, one window size, and a classifier trained on 64 clean images per split. The reported accuracy describes that classifier. Kernel development included severity-3 fog, so the severity-2 result demonstrates sensitivity to a different intensity of a **known corruption family**. Transfer to unfamiliar changes such as lens contamination, motion blur, or sensor noise remains an open question.

The 300 comparisons reuse finite pools drawn from the first 320 test identities. They estimate rejection frequencies for this subset and sampling design. They do not constitute 300 independent deployment episodes. For industrial video, adjacent frames can also be strongly correlated, which affects permutation calibration. A field evaluation would require time-separated reference and evaluation periods, a calibration procedure appropriate for correlated frames, and independent measurements of detection delay, false alarms per operating hour, and downstream classification errors.

## References

1. Gretton et al. _A Kernel Two-Sample Test_. Journal of Machine Learning Research, 2012.
    
2. Rabanser, Günnemann, and Lipton. _Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift_. NeurIPS, 2019.
    
3. Oquab et al. _DINOv2: Learning Robust Visual Features without Supervision_. 2023.
    
4. Hendrycks and Dietterich. _Benchmarking Neural Network Robustness to Common Corruptions and Perturbations_. ICLR, 2019. [CIFAR-10-C data](https://zenodo.org/records/2535967)
    
5. Liu et al. _Learning Deep Kernels for Non-Parametric Two-Sample Tests_. ICML, 2020. [Paper](https://proceedings.mlr.press/v119/liu20m.html)

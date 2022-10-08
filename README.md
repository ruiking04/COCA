# Deep Contrastive One-Class Time Series Anomaly Detection
This repository provides the implementation of the _Deep Contrastive One-Class Time Series Anomaly Detection_ method, called _COCA_ bellow. 

The implementation uses the [Merlion](https://opensource.salesforce.com/Merlion/v1.1.0/tutorials.html) and the [Tsaug](https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html) libraries.

## Abstract
> The accumulation of time-series data and the absence of labels make time-series Anomaly Detection (AD) a selfsupervised deep learning task.
> Single-normality-assumptionbased methods, which reveal only a certain aspect of the whole normality, are incapable of tasks involved with a
> large number of anomalies. Specifically, Contrastive Learning (CL) methods distance negative pairs, many of which consist of both normal 
> samples, thus reducing the AD performance. Existing multi-normality-assumption-based methods are usually two-staged, firstly pre-training 
> through certain tasks whose target may differ from AD, limiting their performance. To overcome the shortcomings, a deep Contrastive One-Class 
> Anomaly detection method of time series (COCA) is proposed by authors, following the normality assumptions of CL and one-class classification. 
> It treats the origin and reconstructed representations as the positive pair of negative-samples-free CL, namely “sequence contrast”. 
> Next, invariance terms and variance terms compose a contrastive one-class loss function in which the loss of the assumptions is optimized 
> by invariance terms simultaneously and the “hypersphere collapse” is prevented by variance terms. In addition, extensive experiments on two 
> real-world time-series datasets show the superior performance of the proposed method achieves state-of-the-art.

Link to arXiv version [here](https://arxiv.org/abs/2207.01472)
## Installation
This code is based on `Python 3.8`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion` and `ts_dataset` as [Merlion](https://github.com/salesforce/Merlion) suggested.

```
git clone https://github.com/salesforce/Merlion.git
cd Merlion
pip install salesforce-merlion
pip install -e Merlion/ts_datasets/
pip install -r requirements.txt
```

## Repository Structure

### `conf`
This directory contains experiment parameters for all models on IOpsCompetition, UCR datasets.

### `models`
Source code of OCSVM, DeepSVDD, CPC, TS-TCC and COCA models.

### `results`
Directory where the experiment results and checkpoint are saved.

## Usage
```
python coca.py --selected_dataset UCR --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --device cuda --seed 2

# COCA Variants
python coca_no_aug.py --selected_dataset IOpsCompetition --device cuda --seed 1
python coca_no_cl.py --selected_dataset IOpsCompetition --device cuda --seed 1
python coca_no_oc.py --selected_dataset IOpsCompetition --device cuda --seed 1
python coca_no_var.py --selected_dataset IOpsCompetition --device cuda --seed 1
python coca_no_view.py --selected_dataset IOpsCompetition --device cuda --seed 1

# Baseline training
# model_name: IsolationForest, RandomCutForest, SpectralResidual, LSTMED, DAGMM, CPC, OCSVM, DeepSVDD
python baseline.py --dataset UCR --model <model_name>  --debug

# TS_TCC_AD training
python ts_tcc_main.py --training_mode self_supervised --selected_dataset IOpsCompetition --device cuda --seed 5
python ts_tcc_main.py --training_mode anomaly_detection --selected_dataset IOpsCompetition --device cuda --seed 5
```

## Disclosure
This implementation is based on [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch), 
[Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch),
[TS-TCC](https://github.com/emadeldeen24/TS-TCC), and [affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py)

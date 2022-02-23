# Deep Contrastive One-Class Time Series Anomaly Detection
This repository provides the implementation of the _Deep Contrastive One-Class Time Series Anomaly Detection_ method, called _COCA_ bellow. 

The implementation uses the [Merlion](https://opensource.salesforce.com/Merlion/v1.1.0/tutorials.html) and the [Tsaug](https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html) libraries.

## Abstract
> Anomaly detection from unlabeled time-series data with temporal dynamics is a very challenging task. Those approaches which do exist are based on a single assumption of normality or require pretraining. They do not fully consider properties of overall normality and are limited by the quality of the representations. This paper proposes a deep Contrastive One-Class Anomaly detection of time series (COCA) from unlabeled data.
> First, to make it easier to distinguish anomalies from normal samples, the original time series data are expanded via distribution augmentation, and then transformed into two different yet correlated views.
> Second, to learn temporal dependencies which are key characteristics of time series, a powerful Seq2Seq model is used in latent space to reconstruct representations of each time step. Last, we propose a contrastive one-class loss function to build a classifier with only one stage, from which an anomaly score is defined.
> Experiments have been carried out on four real-world time-series datasets. The results manifest that COCA leads to a new state of the art in time series anomaly detection. For example, on the AIOps dataset, it raised the revised point-adjusted F1 score by +7.1 percentage points to an accuracy of 51.6.

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
This directory contains experiment parameters for all models on NAB, IOpsCompetition, UCR, SMAP datasets.

### `models`
Source code of OCSVM, DeepSVDD, CPC, TS-TCC and COCA（OC_CL） models.

### `results`
Directory where the experiment results and checkpoint are saved.

## Disclosure
This implementation is based on [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch), [Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch) and [TS-TCC](https://github.com/emadeldeen24/TS-TCC)

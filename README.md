# Deep Contrastive One-Class Time Series Anomaly Detection
This repository provides the implementation of the _Deep Contrastive One-Class Time Series Anomaly Detection_ method, called _COCA_ bellow. 

The implementation uses the [Merlion](https://opensource.salesforce.com/Merlion/v1.1.0/tutorials.html) and the [Tsaug](https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html) libraries.

## Abstract
> The accumulation of time-series data and the absence of labels make time-series Anomaly Detection (AD) a self-supervised deep learning task.
> Single-normality-assumptionbased methods, which reveal only a certain aspect of the whole normality, are incapable of tasks involved with a
> large number of anomalies. Specifically, Contrastive Learning (CL) methods distance negative pairs, many of which consist of both normal 
> samples, thus reducing the AD performance. Existing multi-normality-assumption-based methods are usually two-staged, firstly pre-training 
> through certain tasks whose target may differ from AD, limiting their performance. To overcome the shortcomings, a deep Contrastive One-Class 
> Anomaly detection method of time series (COCA) is proposed by authors, following the normality assumptions of CL and one-class classification. 
> It treats the origin and reconstructed representations as the positive pair of negative-samples-free CL, namely “sequence contrast”. 
> Next, invariance terms and variance terms compose a contrastive one-class loss function in which the loss of the assumptions is optimized 
> by invariance terms simultaneously and the “hypersphere collapse” is prevented by variance terms. In addition, extensive experiments on two 
> real-world time-series datasets show the superior performance of the proposed method achieves state-of-the-art.



## Citation
Link to our paper [here](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch78).

If you use this code for your research, please cite our paper:

```
@inproceedings{wang2023deep,
  title={Deep Contrastive One-Class Time Series Anomaly Detection},
  author={Wang, Rui and Liu, Chongwei and Mou, Xudong and Gao, Kai and Guo, Xiaohui and Liu, Pin and Wo, Tianyu and Liu, Xudong},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={694--702},
  year={2023},
  organization={SIAM}
}
```

## Installation
This code is based on `Python 3.8`, all requirements are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as [Merlion](https://github.com/salesforce/Merlion) suggested.

```
git clone https://github.com/salesforce/Merlion.git
cd Merlion
pip install salesforce-merlion==1.1.1
pip install IPython
pip install -r requirements.txt
```
The COCA repository already includes the merlion's data loading package `ts_datasets`.
Please unzip the `data/iops_competition/phase2.zip` before running the program.

## Repository Structure

### `conf`
This directory contains experiment parameters for all models on IOpsCompetition, UCR datasets.

### `models`
Source code of COCA model.

### `results`
Directory where the experiment results and checkpoint are saved.

## Usage
```
python coca.py --selected_dataset UCR --selected_model COCA --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --selected_model COCA --device cuda --seed 2

# COCA Variants
python coca.py --selected_dataset IOpsCompetition --selected_model COCA_no_aug --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --selected_model COCA_no_cl --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --selected_model COCA_no_oc --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --selected_model COCA_no_var --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --selected_model COCA_view --device cuda --seed 2
```
## Disclosure
This implementation is based on [affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py).

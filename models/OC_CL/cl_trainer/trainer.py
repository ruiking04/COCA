import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from merlion.evaluate.anomaly import TSADMetric, accumulate_tsad_score, TSADScoreAccumulator as ScoreAcc, ScoreType
from merlion.utils import TimeSeries
from utils import label_convert
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

sys.path.append("../../OC_CL")


def Trainer(model, model_optimizer, train_dl, test_dl, train_residual_label, test_residual_label,
            device, logger, config, experiment_log_dir):
    # Start training
    logger.debug("Training started ....")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    all_epoch_train_loss, all_epoch_test_loss = [], []
    all_epoch_test_f1, all_epoch_test_precision, all_epoch_test_recall = [], [], []
    center_train = torch.zeros(config.final_out_channels, device=device)
    center_test = torch.zeros(config.final_out_channels, device=device)
    center_train = center_c2(train_dl, model, device, center_train, config, eps=config.center_eps)
    center_test = center_c2(test_dl, model, device, center_test, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length = model_train(model, model_optimizer, train_dl, center_train,
                                                                    length, config, device, epoch)
        test_target, test_score, test_loss, all_projection = model_evaluate(model, test_dl, center_train, length, config,
                                                                            device, epoch)
        if epoch < config.change_center_epoch:
            center_train = center_c2(train_dl, model, device, center_train, config, eps=config.center_eps)
            center_test = center_c2(train_dl, model, device, center_test, config, eps=config.center_eps)
        scheduler.step(train_loss)
        # according to scores to create predicting labels
        test_score, _ = ad_predict1(test_target, test_score, config)

        test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
        test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
        test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \n'
                     f'test Loss     : {test_loss:.4f}\t  | \ttest F1     : {test_f1:2.4f}  | \ttest precision     : {test_precision:2.4f}  | \ttest recall     : {test_recall:2.4f}\n')

        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(test_loss.item())

        all_epoch_test_f1.append(test_f1)
        all_epoch_test_precision.append(test_precision)
        all_epoch_test_recall.append(test_recall)

    # visualization
    # test_target_copy = torch.Tensor(test_target_copy).unsqueeze(1)
    # test_target_copy = torch.cat((test_target_copy, test_target_copy, test_target_copy), 1).reshape(-1)
    # visual_target = np.zeros(len(test_target_copy))
    # test_predict_copy = torch.Tensor(test_predict_copy).unsqueeze(1)
    # visual_predict = torch.cat((test_predict_copy, test_predict_copy, test_predict_copy), 1).reshape(-1)
    # # visual_prototypes = model.prototypes_layer.weight.data.clone()
    # for inx, value in enumerate(test_target_copy):
    #     if value == 1:
    #         visual_target[inx] = -1
    #     else:
    #         visual_target[inx] = 1
    # embedding_target = visual_predict + visual_target
    # index1 = np.where(embedding_target == -1)[0][:200]
    # index2 = np.where(embedding_target == 0)[0][:200]
    # index3 = np.where(embedding_target == 1)[0][:200]
    # index4 = np.where(embedding_target == 2)[0][:200]
    # index5 = np.hstack((index1, index2, index3, index4))
    # part_embedding_target = embedding_target[index5]
    # part_embedding_feature = all_projection[index5]
    # part_embedding_feature = torch.cat((part_embedding_feature, center.reshape(1, -1)), dim=0)
    # # part_embedding_feature = F.normalize(part_embedding_feature, dim=1)
    # part_embedding_target = np.append(part_embedding_target, 3)

    # writer = SummaryWriter()
    # for i in range(config.num_epoch):
    #     writer.add_scalars('loss', {'train': all_epoch_train_loss[i],
    #                                 'test': all_epoch_test_loss[i]}, i)
    #     writer.add_scalars('f1', {'train': all_epoch_train_f1[i],
    #                               'test': all_epoch_test_f1[i]}, i)
    #     writer.add_scalars('precision', {'train': all_epoch_train_precision[i],
    #                                      'test': all_epoch_test_precision[i]}, i)
    #     writer.add_scalars('recall', {'train': all_epoch_train_recall[i],
    #                                   'test': all_epoch_test_recall[i]}, i)
    # # writer.add_embedding(part_embedding_feature, metadata=part_embedding_target, tag='test embedding')
    # writer.close()

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    logger.debug("\n################## Training is Done! #########################")
    return train_score, test_score
    # return all_train_target, all_train_predict, all_test_target, all_test_predict


def model_train(model, model_optimizer, train_loader, center, length, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()

        feature1, feature_dec1 = model(aug1)
        feature2, feature_dec2 = model(aug2)
        loss, score1, score2 = train(feature1, feature2, feature_dec1, feature_dec2, center, length, epoch, config, device)
        all_score = torch.cat((score1.unsqueeze(1), score2.unsqueeze(1)), dim=1)
        # Update hypersphere radius R on mini-batch distances
        if (config.objective == 'soft-boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(all_score, config.detect_nu), device=device)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

        target = target.reshape(-1)

        predict = all_score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss, length


def model_evaluate(model, test_dl, center, length, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    all_projection = []
    with torch.no_grad():
        for data, target, aug1, aug2 in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

            feature1, pred1 = model(aug1)
            feature2, pred2 = model(aug2)
            # center_test = center
            loss, score1, score2 = train(feature1, feature2, pred1, pred2, center, length, epoch, config, device)
            total_loss.append(loss.item())
            all_score = torch.cat((score1.unsqueeze(1), score2.unsqueeze(1)), dim=1)
            predict = all_score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.cpu())
            all_predict.extend(predict)
            all_projection.append(feature1)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_projection = torch.cat(all_projection, dim=0)

    return all_target, all_predict, total_loss, all_projection


def train(feature1, feature2, feature_dec1, feature_dec2, center, length, epoch, config, device):
    # normalize feature vectors
    center = center.unsqueeze(0)
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)
    feature_dec1 = F.normalize(feature_dec1, dim=1)
    feature_dec2 = F.normalize(feature_dec2, dim=1)
    center = F.normalize(center, dim=1)

    distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
    distance2 = F.cosine_similarity(feature2, center, eps=1e-6)
    distance_dec1 = F.cosine_similarity(feature_dec1, center, eps=1e-6)
    distance_dec2 = F.cosine_similarity(feature_dec2, center, eps=1e-6)
    distance1 = 1 - distance1
    distance2 = 1 - distance2
    distance_dec1 = 1 - distance_dec1
    distance_dec2 = 1 - distance_dec2

    # The radian loss is calculated across the representations of the four views and reconstruction
    if config.loss_type == 'arc1':
        cos1 = F.cosine_similarity(feature1 - center, feature_dec2 - center, eps=1e-6)
        cos2 = F.cosine_similarity(feature2 - center, feature_dec1 - center, eps=1e-6)
        # Prevents gradients from appearing NAN
        cos1[cos1 < 0] += 0.001
        cos1[cos1 > 0] -= 0.001
        cos2[cos2 < 0] += 0.001
        cos2[cos2 > 0] -= 0.001
        theta1 = torch.acos(cos1)
        theta2 = torch.acos(cos2)
        score1 = (distance1 + distance_dec2) * theta1 / 2
        score2 = (distance2 + distance_dec1) * theta2 / 2
        if config.objective == 'soft-boundary':
            diff1 = score1 ** 2 - length ** 2
            diff2 = score2 ** 2 - length ** 2
            loss = length ** 2 + (1 / config.nu) * (torch.mean(torch.max(torch.zeros_like(diff1), diff1)) + torch.mean(
                   torch.max(torch.zeros_like(diff2), diff2))) / 2
        else:
            loss = torch.mean(0.5 * (score1 + score2))

    # The radian losses are calculated for the four views and reconstructed representations
    elif config.loss_type == 'arc2':
        cos1 = F.cosine_similarity(feature1 - center, feature_dec1 - center, eps=1e-6)
        cos2 = F.cosine_similarity(feature2 - center, feature_dec2 - center, eps=1e-6)
        cos3 = F.cosine_similarity(feature1 - center, feature2 - center, eps=1e-6)
        # Prevents gradients from appearing NAN
        cos1[cos1 < 0] += 0.001
        cos1[cos1 > 0] -= 0.001
        cos2[cos2 < 0] += 0.001
        cos2[cos2 > 0] -= 0.001
        cos3[cos3 < 0] += 0.001
        cos3[cos3 > 0] -= 0.001

        theta1 = torch.acos(cos1)
        theta2 = torch.acos(cos2)
        theta3 = torch.acos(cos2)
        score1 = (distance1 + distance_dec1) * theta1 / 2
        score2 = (distance2 + distance_dec2) * theta2 / 2
        score3 = (distance1 + distance2) * theta3 / 2
        score = score1 + score2 + score3
        if config.objective == 'soft-boundary':
            diff = score ** 2 - length ** 2
            loss = length ** 2 + (1 / config.nu) * (torch.mean(torch.max(torch.zeros_like(diff), diff))) / 2
        else:
            loss = torch.mean(0.5 * score)
        score1 = score
        score2 = score

    # The loss function used in ablation experiments
    elif config.loss_type == 'mix':
        score = distance1 + distance_dec1 + distance2 + distance_dec2
        if config.objective == 'soft-boundary':
            diff = score ** 2 - length ** 2
            loss = length ** 2 + (1 / config.nu) * (torch.mean(torch.max(torch.zeros_like(diff), diff))) / 2
        else:
            loss = torch.mean(score)
        score1 = score
        score2 = score

    # The loss function used in ablation experiments
    elif config.loss_type == 'no_reconstruction':
        score = distance1 + distance2
        if config.objective == 'soft-boundary':
            diff = score ** 2 - length ** 2
            loss = length ** 2 + (1 / config.nu) * (torch.mean(torch.max(torch.zeros_like(diff), diff))) / 2
        else:
            loss = torch.mean(score)
            # loss = score
        score1 = score
        score2 = score

    # The loss function used in our paper
    else:
        score1 = 2 * distance1 + distance_dec1 + distance2
        score2 = 2 * distance2 + distance_dec2 + distance1
        if config.objective == 'soft-boundary':
            # diff = score ** 2 - length ** 2
            # loss = length ** 2 + (1 / config.nu) * (torch.mean(torch.max(torch.zeros_like(diff), diff))) / 2
            diff1 = score1 ** 2 - length ** 2
            diff2 = score2 ** 2 - length ** 2
            loss = length ** 2 + (1 / config.nu) * (torch.mean(torch.max(torch.zeros_like(diff1), diff1)) + torch.mean(
                   torch.max(torch.zeros_like(diff2), diff2))) / 2
        else:
            loss = torch.mean(0.5 * (score1 + score2))
    return loss, score1, score2


@torch.no_grad()
def ad_predict(target, scores, nu, config):
    sequence_length = config.features_len
    scores = np.array(scores)
    lattice = np.max(scores, axis=1)
    lattice = lattice.reshape(-1, sequence_length)
    lattice = np.max(lattice, axis=1)
    scores = lattice
    predict = np.zeros(len(scores))
    mean = np.mean(scores)
    std = np.std(scores)
    scores = (scores - mean)/std
    threshold = np.percentile(scores, nu)
    mount = 0
    for index, r2 in enumerate(scores):
        if r2.item() > threshold:
            predict[index] = 1
            mount += 1
    return target, predict


@torch.no_grad()
def ad_predict1(target, scores, config):
    target = TimeSeries.from_pd(pd.DataFrame(target))
    sequence_length = config.features_len
    scores = np.array(scores)
    # Find the maximum anomaly score in two views
    lattice = np.max(scores, axis=1)
    # Find the maximum anomaly score in a sequence
    lattice = lattice.reshape(-1, sequence_length)
    lattice = np.max(lattice, axis=1)
    scores = lattice

    # z-score standardize
    mean = np.mean(scores)
    std = np.std(scores)
    if std != 0:
        scores = (scores - mean)/std

    # For UCR dataset, the evaluation just checks whether the point with the highest anomaly score is anomalous or not.
    # For UCR dataset, there is only one anomaly period in the test set.
    if config.threshold_determine == 'one-anomaly':
        mount = 0
        threshold = np.max(scores, axis=0)
        # scores_sort = np.argsort(scores, axis=0)
        # threshold = scores[scores_sort[-9]]
        predict = np.zeros(len(scores))
        for index, r2 in enumerate(scores):
            if r2.item() >= threshold:
                predict[index] = 1
                mount += 1
        print(mount)
        predict = TimeSeries.from_pd(pd.DataFrame(predict))
        score_max = accumulate_tsad_score(ground_truth=target, predict=predict)
        # f1_max = score_max.f1(ScoreType.RevisedPointAdjusted)
        n_anom = score_max.num_tp_anom + score_max.num_fn_anom
        if n_anom == 0:
            score_max.num_tp_anom, score_max.num_fn_anom, score_max.num_fp = 0, 0, 0
        elif score_max.num_tp_anom > 0:
            score_max.num_tp_anom, score_max.num_fn_anom, score_max.num_fp = 1, 0, 0
        else:
            score_max.num_tp_anom, score_max.num_fn_anom, score_max.num_fp = 0, 1, 1
        nu_max = 1

    # Fixed threshold
    elif config.threshold_determine == 'fix':
        detect_nu = 100 * (1 - config.detect_nu)
        threshold = np.percentile(scores, detect_nu)
        mount = 0
        predict = np.zeros(len(scores))
        for index, r2 in enumerate(scores):
            if r2.item() > threshold:
                predict[index] = 1
                mount += 1
        predict = TimeSeries.from_pd(pd.DataFrame(predict))
        score_max = accumulate_tsad_score(ground_truth=target, predict=predict)
        nu_max = detect_nu

    # Floating threshold
    else:
        nu_list = np.arange(1, 301) / 1e3
        f1_list, score_list = [], []
        for detect_nu in nu_list:
            threshold = np.percentile(scores, 100-detect_nu)
            mount = 0
            predict = np.zeros(len(scores))
            for index, r2 in enumerate(scores):
                if r2.item() > threshold:
                    predict[index] = 1
                    mount += 1
            predict = TimeSeries.from_pd(pd.DataFrame(predict))
            score = accumulate_tsad_score(ground_truth=target, predict=predict)
            f1 = score.f1(ScoreType.RevisedPointAdjusted)
            f1_list.append(f1)
            score_list.append(score)

        index_max = np.argmax(f1_list, axis=0)
        # print(f1_list)
        # print(f1_list[index_max])
        score_max = score_list[index_max]
        nu_max = nu_list[index_max]
        # print(nu_max)
    return score_max, nu_max


@torch.no_grad()
def ad_evaluate(predict, target):
    target = target.reshape(-1)
    target = target.detach().numpy()
    f1 = TSADMetric.F1.value(ground_truth=TimeSeries.from_pd(target),
                                 predict=TimeSeries.from_pd(predict))
    p = TSADMetric.Precision.value(ground_truth=TimeSeries.from_pd(target),
                                       predict=TimeSeries.from_pd(predict))
    r = TSADMetric.Recall.value(ground_truth=TimeSeries.from_pd(target),
                                    predict=TimeSeries.from_pd(predict))
    return f1, p, r


def center_c(center, feature1, feature2, feature_dec1, feature_dec2, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    c = center
    beta = config.center_beta
    # all_feature = torch.cat((feature1, feature2, feature_dec1, feature_dec2), dim=0)
    all_feature = torch.cat((feature1, feature2), dim=0)
    update = torch.mean(all_feature, dim=0)
    if np.count_nonzero(c.detach().cpu().numpy()) == 0:
        beta = 0
    # if torch.count_nonzero(c) == 0:
    #     beta = 0
    update_c = beta * c + (1 - beta) * update
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    update_c[(abs(update_c) < eps) & (update_c < 0)] = -eps
    update_c[(abs(update_c) < eps) & (update_c > 0)] = eps
    return update_c


def center_c2(train_loader, model, device, center, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            _, _, aug1, aug2 = data
            aug1 = aug1.float().to(device)
            aug2 = aug2.float().to(device)
            outputs1, dec1 = model(aug1)
            outputs2, dec2 = model(aug2)
            n_samples += outputs1.shape[0]
            if config.loss_type == 'no_reconstruction':
                all_feature = torch.cat((outputs1, outputs2), dim=0)
            else:
                all_feature = torch.cat((outputs1, outputs2, dec1, dec2), dim=0)
            c += torch.sum(all_feature, dim=0)

    c /= (4 * n_samples)

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)




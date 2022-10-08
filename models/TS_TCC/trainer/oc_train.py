import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator

def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, logger, config):
    # Start training
    logger.debug("one-class classfication fine-tune started ....")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    all_epoch_train_loss, all_epoch_test_loss = [], []
    center = torch.zeros(config.final_out_channels, device=device)
    center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length = model_train(model, model_optimizer, train_dl, center,
                                                                    length, config, device, epoch)
        val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl, center, length, config,
                                                                                device, epoch)
        test_target, test_score_origin, test_loss, all_projection = model_evaluate(model, test_dl, center, length,
                                                                                   config, device, epoch)
        scheduler.step(train_loss)
        # logger.debug(f'\nEpoch : {epoch}\n'
        #              f'Train Loss     : {train_loss:.4f}\t | \n'
        #              f'Valid Loss     : {val_loss:.4f}\t  | \n'
        #              f'Test Loss     : {test_loss:.4f}\t  | \n'
        #              )
        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(val_loss.item())

    # according to scores to create predicting labels
    val_affiliation, val_score, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                               config.detect_nu)
    test_affiliation, test_score, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                       config.detect_nu)
    if config.dataset == 'UCR':
        score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
        print("Test accuracy metrics")
        logger.debug(
            f'Test accuracy: {score_reasonable.correct_num:2.4f}\n')
    else:
        score_reasonable = reasonable_accumulator(1, 0)

    logger.debug("\n################## Training is Done! #########################")
    val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
    val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
    val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
    print("Valid affiliation-metrics")
    logger.debug(
        f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
    print("Valid RAP F1")
    logger.debug(
        f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')

    test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
    test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
    test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test affiliation-metrics")
    logger.debug(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    print("Test RAP F1")
    logger.debug(
        f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')

    return test_score_origin, test_affiliation, test_score, score_reasonable, predict


def model_train(model, model_optimizer, train_loader, center, length, config, device, epoch):
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        # optimizer
        model_optimizer.zero_grad()
        _, features = model(data)
        features = features.permute(0, 2, 1)
        features = features.reshape(-1, config.final_out_channels)
        loss, score = train(features, center, length, epoch, config, device)
        # Update hypersphere radius R on mini-batch distances
        if (config.objective == 'soft-boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(score, config.nu), device=device)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

        target = target.reshape(-1)

        predict = score.detach().cpu().numpy()
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
            _, features = model(data)
            features = features.permute(0, 2, 1)
            features = features.reshape(-1, config.final_out_channels)
            loss, score = train(features, center, length, epoch, config, device)
            total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)
            all_projection.append(features)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_projection = torch.cat(all_projection, dim=0)
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss, all_projection


def train(feature1, center, length, epoch, config, device):
    # normalize feature vectors
    center = center.unsqueeze(0)
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
    distance1 = 1 - distance1

    # The Loss function that representations reconstruction
    score = distance1
    if config.objective == 'soft-boundary':
        diff1 = score - length
        loss_oc = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
    else:
        loss_oc = torch.mean(score)
    loss = loss_oc
    return loss, score


def center_c(train_loader, model, device, center, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data, target, aug1, aug2 = data
            data = data.float().to(device)
            _, features = model(data)
            features = features.permute(0, 2, 1)
            features = features.reshape(-1, config.final_out_channels)
            n_samples += features.shape[0]
            c += torch.sum(features, dim=0)

    c /= (n_samples)

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)
from copy import deepcopy
import logging
import os
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from classifier import Classifier
from image_list import ImageList
from moco.builder import AdaMoCo
from moco.loader import NCropsTransform
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
)

CE = nn.CrossEntropyLoss(reduction='none')

@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, args):
    wandb_dict = dict()

    # make sure to switch to eval mode
    model.eval()

    # run inference
    logits, gt_labels, indices = [], [], []
    features = []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")

        # (B, D) x (D, K) -> (B, K)
        feats, logits_cls = model(imgs, cls_only=True)

        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)

    features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices = torch.cat(indices).to("cuda")

    if args.distributed:
        # gather results from all ranks
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        # remove extra wrap-arounds from DDP
        ranks = len(dataloader.dataset) % dist.get_world_size()
        features = remove_wrap_arounds(features, ranks)
        logits = remove_wrap_arounds(logits, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logging.info(f"Accuracy of direct prediction: {accuracy:.2f}")
    wandb_dict["Test Acc"] = accuracy
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict["Test Avg"] = acc_per_class.mean()
        wandb_dict["Test Per-class"] = acc_per_class

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.learn.queue_size],
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, acc = refine_predictions(
        features, probs, banks, args=args, gt_labels=gt_labels
    )
    wandb_dict["Test Post Acc"] = acc
    if args.data.dataset == "VISDA-C":
        acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
        wandb_dict["Test Post Avg"] = acc_per_class.mean()
        wandb_dict["Test Post Per-class"] = acc_per_class

    pseudo_item_list = []
    for pred_label, idx in zip(pred_labels, indices):
        img_path, _, mask_path, img_file = dataloader.dataset.item_list[idx]
        pseudo_item_list.append((img_path, int(pred_label), mask_path, img_file))
    logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")

    if use_wandb(args):
        wandb.log(wandb_dict)

    return pseudo_item_list, banks, accuracy


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


@torch.no_grad()
def update_labels(banks, idxs, features, logits, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)

    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])


@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
    elif args.learn.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.learn.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy


def get_augmentation_versions(args):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in args.learn.aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(args.data.aug_type, args.learn.alpha_spm, args.learn.beta_spm, args.learn.patch_height, args.learn.mix_prob))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "n":
            transform_list.append(get_augmentation("jigsaw"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    if args.learn.negative_aug:
        transform_list.append(get_augmentation("jigsaw",patch_height=args.learn.patch_height))
    transform = NCropsTransform(transform_list)

    return transform


def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": args.optim.lr * 10,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer

def preparetrainloader(args, pseudo_item_list):
    # Training data
    train_transform = get_augmentation_versions(args)
    train_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=None,  # uses pseudo labels
        transform=train_transform,
        pseudo_item_list=pseudo_item_list,
    )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.data.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    return train_loader, train_sampler

def train_target_domain(args):
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )
    
    # 
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    os.chdir(parent_dir)
    
    # if not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        label_file = os.path.join(
            args.data.image_root, f"{args.data.tgt_domain}_list.txt"
        )
        dummy_dataset = ImageList(args.data.image_root, label_file)
        data_length = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset

    checkpoint_path = os.path.join(
        args.model_tta.src_log_dir,
        f"best_{args.data.src_domain}_{args.seed}.pth.tar",
    )
    src_model = Classifier(args.model_src, checkpoint_path)
    momentum_model = Classifier(args.model_src, checkpoint_path)
    model = AdaMoCo(
        src_model,
        momentum_model,
        K=args.model_tta.queue_size,
        m=args.model_tta.m,
        T_moco=args.model_tta.T_moco,
        negative_aug=args.learn.negative_aug,
    ).cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu])
    logging.info(f"1 - Created target model")

    val_transform = get_augmentation("test")
    label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    val_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,
        transform=val_transform,
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, sampler=val_sampler, num_workers=2
    )
    pseudo_item_list, banks, _ = eval_and_label_dataset(
        val_loader, model, banks=None, args=args
    )
    logging.info("2 - Computed initial pseudo labels")

    train_loader, train_sampler = preparetrainloader(args, pseudo_item_list)

    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info("3 - Created train/val loader")

    # define loss function (criterion) and optimizer
    optimizer = get_target_optimizer(model, args)
    logging.info("4 - Created optimizer")

    logging.info("Start training...")
    start_alpha = args.learn.alpha_spm
    end_alpha = args.learn.alpha_spm_end

    best_acc = 0

    # Makes sure the code runs for minimum of 3K Iterations
    if args.learn.iterations > args.learn.epochs * (len(train_loader)):
        args.learn.epochs = int(args.learn.iterations / len(train_loader))
    logging.info(f"5 - Number of Epochs = {args.learn.epochs}")

    for epoch in range(args.learn.start_epoch, args.learn.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, banks, optimizer, epoch, args)

        _, _, accuracy = eval_and_label_dataset(val_loader, model, banks, args)
        
        if accuracy>best_acc:  
            if is_master(args):
                best_acc = accuracy
                filename = f"checkpoint_best_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
                save_path_best = os.path.join(args.log_dir, filename)
                save_checkpoint(model, optimizer, epoch, save_path=save_path_best)
                logging.info(f"Saved checkpoint {save_path_best}")

        if args.learn.change_alpha:
            args.learn.alpha_spm = round(args.learn.alpha_spm - (start_alpha-end_alpha)/(args.learn.epochs),2)
            if args.learn.alpha_spm < 2:
                args.learn.alpha_spm = 2
            train_loader, train_sampler = preparetrainloader(args, pseudo_item_list)
        logging.info(f'Alpha Changed to {args.learn.alpha_spm}')

    if is_master(args):
        filename = f"checkpoint_{epoch:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
        save_path = os.path.join(args.log_dir, filename)
        save_checkpoint(model, optimizer, epoch, save_path=save_path)
        logging.info(f"Saved checkpoint {save_path}")

    if args.data.ttd:
        # Load the best model 
        checkpoint = torch.load(save_path_best)
        model.load_state_dict(checkpoint['state_dict'])

        # evaluate on specific target domains
        for t, tgt_domain in enumerate(args.data.test_target_domain):
            if tgt_domain == args.data.src_domain:
                continue
            label_file = os.path.join(args.data.image_root, f"{tgt_domain}_list.txt")
            tgt_dataset = ImageList(args.data.image_root, label_file, val_transform)
            sampler = DistributedSampler(tgt_dataset, shuffle=False) if args.distributed else None
            tgt_loader = DataLoader(
                tgt_dataset,
                batch_size=args.data.batch_size,
                sampler=sampler,
                pin_memory=True,
                num_workers=args.data.workers,
            )

            logging.info(f"Evaluate {args.data.src_domain} model on {tgt_domain}")
            logging.info(f"Dataset length: {len(tgt_loader.dataset)}")
            
            eval_and_label_dataset(tgt_loader, model, banks, args)

def update_w(iteration, w_in, args):
    schedule_start, schedule_end = args.learn.schedule
    scale = 0

    if iteration <= schedule_start:
        scale = 0
    elif iteration > schedule_end:
        scale = 1
    else:
        # Calculate step decay directly
        total_steps = (schedule_end - schedule_start) // 10
        step_size = 1 / total_steps
        scale = ((iteration - schedule_start) // 10) * step_size

    return w_in * scale + (1 - scale)


def train_epoch(train_loader, model, banks, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_ins = AverageMeter("SSL-Acc@1", ":6.2f")
    top1_psd = AverageMeter("CLS-Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    # make sure to switch to train mode
    model.train()
    torch.autograd.set_detect_anomaly(True)

    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    for i, data in enumerate(train_loader):
        # unpack and move data
        images, _, idxs = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )

        if args.learn.negative_aug:
            images_n = images[3].to("cuda")
        else:
            images_n = None

        # per-step scheduler
        if args.optim.lr_decay:
            step = i + epoch * len(train_loader)
            adjust_learning_rate(optimizer, step, args)

        feats_w, logits_w = model(images_w, cls_only=True)

        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args
            )

        _, logits_q, logits_ins, keys, keys_neg = model(images_q, images_k, images_n)
        # update key features and corresponding pseudo labels
        model.update_memory(keys, pseudo_labels_w, keys_neg)

        # moco instance discrimination
        loss_ins, accuracy_ins = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=model.mem_labels,
            args=args,
        )
        # instance accuracy shown for only one process to give a rough idea
        top1_ins.update(accuracy_ins.item(), len(logits_ins))

        # classification
        '''
        loss_cls, accuracy_psd = classification_loss(
            logits_w, logits_q, pseudo_labels_w, args
        )
        '''

        if args.learn.reweighting:
            with torch.no_grad():
                w = confidence_margin_reweighting(probs_w,args)
                # w = entropy_reweighting(probs_w)
                step = i + epoch * len(train_loader)
                w = update_w(step, w, args)
                if i==0:
                    None
                    # visualize_confidence_margin(probs_w,epoch,args)
            loss_cls = (w * CE(logits_q, pseudo_labels_w)).mean()
        else:
            loss_cls = (CE(logits_q, pseudo_labels_w)).mean()

        accuracy_psd = calculate_acc(logits_q, pseudo_labels_w)
        top1_psd.update(accuracy_psd.item(), len(logits_w))

        # diversification
        loss_div = (
            diversification_loss(logits_w, logits_q, args)
        )
        
        # Increase lambda_cls from 0 to 1 using step decay
        # lambda_cls = update_lambda_cls(step, args)

        loss = (
            args.learn.lambda_cls * loss_cls
            + args.learn.lambda_ins * loss_ins
            + args.learn.lambda_div * loss_div
        )
        
        loss_meter.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, args)

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.lambda_cls * loss_cls.item(),
                "loss_ins": args.learn.lambda_ins * loss_ins.item(),
                "loss_div": args.learn.lambda_div * loss_div.item(),
                "acc_ins": accuracy_ins.item(),
                "acc_pseudo-labels": accuracy_psd.item(),
            }

            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)


@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy


def instance_loss(logits_ins, pseudo_labels, mem_labels, args):
    contrast_type = args.learn.contrast_type
    k = args.model_tta.queue_size
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:k+1] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        mask[:, k:] = False
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)
    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy


def classification_loss(logits_w, logits_s, target_labels, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels, args)
        accuracy = calculate_acc(logits_w, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_cls = cross_entropy_loss(logits_s, target_labels, args)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div


def diversification_loss(logits_w, logits_s, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels, args):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss


def visualize_confidence_margin(probs, epoch, args):
    # Ensure the visualize directory exists
    visualize_dir = os.path.join(args.log_dir, 'visualize')
    os.makedirs(visualize_dir, exist_ok=True)

    with torch.no_grad():
        # Sort probabilities to get top 2 probabilities for each sample
        top_probs, _ = torch.topk(probs, k=2, dim=1)
        confidence = top_probs[:, 0]  # Top confidence value
        margin = confidence - top_probs[:, 1]  # Confidence margin

    # Prepare data for plotting
    confidence_np = confidence.cpu().numpy()
    margin_np = margin.cpu().numpy()

    # Create the joint plot
    plt.figure(figsize=(8, 6))
    joint_plot = sns.jointplot(x=confidence_np, y=margin_np, kind="scatter", color="blue")
    joint_plot.ax_joint.set_xlim(0, 1)  # Set x-axis limit
    joint_plot.ax_joint.set_ylim(0, 1)  # Set y-axis limit

    # Add labels and title
    joint_plot.set_axis_labels("Confidence", "Margin")
    plt.suptitle(f"Joint Distribution of Confidence and Margin (Epoch {epoch})", y=1.02)

    # Save the plot
    filename = f"{epoch}.png"
    save_path = os.path.join(visualize_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")


def confidence_margin_reweighting(probs,args):
    with torch.no_grad():
        # Sort probabilities to get top 2 probabilities for each sample
        top_probs, _ = torch.topk(probs, k=2, dim=1)
        c = top_probs[:, 0]                     # Compute confidence 
        m = top_probs[:, 0] - top_probs[:, 1]   # Compute margin
        cm_p = (c + m)/2
        cm = c*m
        
        # Normalize to [0, 1] 
        m_norm = m / torch.max(m) 
        c_norm = c / torch.max(c) 
        cm_norm = cm / torch.max(cm) 
        
        # Compute reweighting factors based on selected strategy
        reweighting_type = args.learn.reweighting_type
        
        # Define reweighting strategy mapping
        reweighting_map = {
            "m": m, "c": c, "cm": cm, "cm_p": cm_p,
            "m_2": m ** 2, "c_2": c ** 2, "cm_2": cm ** 2, "cm_p_2": cm_p ** 2,
            "exp_m": torch.exp(m), "exp_c": torch.exp(c), "exp_cm": torch.exp(cm), "exp_cm_p": torch.exp(cm_p),
            "cm_exp_m":cm * torch.exp(m),	"cm_exp_c":cm * torch.exp(c),	"cm_p_exp_m":cm_p * torch.exp(m),	"cm_p_exp_c":cm_p * torch.exp(c),
            "m_exp_c":m * torch.exp(c), "c_exp_m":c * torch.exp(m),
            "cm_m_2": cm * (m ** 2), "cm_m_3": cm * (m ** 3), "cm_c_2": cm * (c ** 2), "cm_c_3": cm * (c ** 3), 
            "2_c_m_2": 2 * c * (m ** 2), "3_c_m_3": 3 * c * (m ** 3), "2_m_c_2": 2 * m * (c ** 2), "3_m_c_3": 3 * m * (c ** 3), 
            "2_cm_m_2": 2 * cm * (m ** 2), "3_cm_m_3": 3 * cm * (m ** 3), "2_cm_c_2": 2 * cm * (c ** 2), "3_cm_c_3": 3 * cm * (c ** 3), 
            "2_cm":2 * cm, "3_m_c_2":3 * m * (c**2)	, "3_c_m_2":3 * c * (m**2)	, "4_c_m_3":4 * c * (m ** 3)	, "4_m_c_3":4 * m * (c ** 3),
            #
            "cm_3": cm ** 3, "m_norm": m_norm, "c_norm": c_norm, "cm_norm": cm_norm,
            "cm_exp_m": cm * torch.exp(m), "cm_exp_m_norm": cm * torch.exp(m_norm),
            "cm_exp_c": cm * torch.exp(c), "cm_exp_c_norm": cm * torch.exp(c_norm),
            "m_exp_c": m * torch.exp(c), "c_exp_m": c * torch.exp(m),
            "m_c2": m * c ** 2, "c_m2": c * m ** 2, "m_c3": m * c ** 3, "c_m3": c * m ** 3,
            "cm_c2": cm * c ** 2, "cm_m2": cm * m ** 2, "cm_c3": cm * c ** 3, "cm_m3": cm * m ** 3
        }
            
        # Get weights or raise an error if the type is not recognized
        weights = reweighting_map.get(reweighting_type)
        if weights is None:
            raise ValueError(f"Reweighting type '{reweighting_type}' not implemented")

    return weights


def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)


def entropy_reweighting(probs):
        num_classes = probs.shape[1]
        with torch.no_grad():
            #CE weights
            max_entropy = torch.log2(torch.tensor(num_classes, dtype=torch.float32))
            weight = torch.exp( - entropy(probs) / max_entropy)
        return weight
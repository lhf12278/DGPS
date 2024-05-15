import torch


def make_optimizer(cfg, model, center_criterion):
    """ Update the parameters of the classifier"""
    params = []
    keys = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bottleneck_proj" in key:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
            continue
        elif "proj_classifier" in key:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
            continue


    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_1(cfg, model, center_criterion):
    """Update the learnable_person_info"""
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "learn_parameters.learnable_person_info" in key:
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_2(cfg, model):
    params = []
    model_params = []
    keys = []
    for key, value in model.named_parameters():
        lr = cfg.SOLVER.tuning_LR
        weight_decay = 0.0001

        if not value.requires_grad:
            continue

        if "learn_parameters.learnable_person_info" in key:
            value.requires_grad_(False)
            continue

        if "learn_parameters.Disturbance_Domain_Data" in key:
            params += [{"params": [value], "lr": cfg.SOLVER.BASE_LR, "weight_decay": weight_decay}]
            keys += [key]
            continue

        model_params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_model = torch.optim.Adam(model_params)

    return optimizer, optimizer_model
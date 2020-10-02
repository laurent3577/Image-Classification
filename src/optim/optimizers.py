from torch import optim


def build_opt(config, model, steps_per_epoch):
    lr = config.OPTIM.BASE_LR
    weight_decay = config.OPTIM.WEIGHT_DECAY
    if config.OPTIM.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif config.OPTIM.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True
        )
    elif config.OPTIM.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("{} unknown optimizer type".format(config.OPTIM.OPTIMIZER))

    scheduler = build_lr_scheduler(optimizer, config, steps_per_epoch)
    return optimizer, scheduler

def build_lr_scheduler(optimizer, config, steps_per_epoch):
    if config.OPTIM.SCHEDULER.TYPE == "Step":
        scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.OPTIM.SCHEDULER.STEP_SIZE,
                gamma=config.OPTIM.SCHEDULER.GAMMA)
        scheduler.update_on_step = False
    elif config.OPTIM.SCHEDULER.TYPE == "Cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.OPTIM.EPOCH*steps_per_epoch,
                eta_min=config.OPTIM.SCHEDULER.COSINE_LR_MIN)
        scheduler.update_on_step = True
    elif config.OPTIM.SCHEDULER.TYPE == "Exp":
        scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.OPTIM.SCHEDULER.GAMMA)
        scheduler.update_on_step = True
    elif config.OPTIM.SCHEDULER.TYPE == "OneCycle":
        # For OneCycle scheduler:
        # max_lr = base_lr * gamma
        # initial_lr = max_lr / div_factor
        # final_lr = initial_lr / 1e4
        scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[pg['lr']*config.OPTIM.SCHEDULER.GAMMA for pg in optimizer.param_groups],
                total_steps=None,
                epochs=config.OPTIM.EPOCH,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=config.OPTIM.SCHEDULER.CYCLE_DIV_FACTOR,
                final_div_factor=1e4)
        scheduler.update_on_step = True
    else:
        raise ValueError("{} unknown scheduler type".format(config.OPTIM.SCHEDULER.TYPE))
    return scheduler
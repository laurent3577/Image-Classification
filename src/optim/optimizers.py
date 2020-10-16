from torch import optim


def build_opt(
    param_groups,
    optimizer_name,
    base_lr,
    weight_decay=1e-5,
    scheduler_name="OneCycle",
    step_size=10,
    gamma=10.0,
    cosine_lr_min=1e-4,
    cycle_div_factor=25,
    epochs=20,
    steps_per_epoch=800,
):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            param_groups, lr=base_lr, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            param_groups, lr=base_lr, weight_decay=weight_decay
        )
    else:
        raise ValueError("{} unknown optimizer type".format(optimizer_name))

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name,
        step_size,
        gamma,
        cosine_lr_min,
        cycle_div_factor,
        epochs,
        steps_per_epoch,
    )
    return optimizer, scheduler


def build_lr_scheduler(
    optimizer,
    scheduler_name,
    step_size,
    gamma,
    cosine_lr_min,
    cycle_div_factor,
    epochs,
    steps_per_epoch,
):
    if scheduler_name == "Step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        scheduler.update_on_step = False
    elif scheduler_name == "Cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch, eta_min=cosine_lr_min
        )
        scheduler.update_on_step = True
    elif scheduler_name == "Exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler.update_on_step = True
    elif scheduler_name == "OneCycle":
        # For OneCycle scheduler:
        # max_lr = base_lr * gamma
        # initial_lr = max_lr / div_factor
        # final_lr = initial_lr / 1e4
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] * gamma for pg in optimizer.param_groups],
            total_steps=None,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=cycle_div_factor,
            final_div_factor=1e4,
        )
        scheduler.update_on_step = True
    else:
        raise ValueError("{} unknown scheduler type".format(scheduler_name))
    return scheduler

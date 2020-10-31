# Pytorch trainer

Implements various wrappers on pytorch for training DL models.

## Features

The main features include:

### Trainer

The trainer class handles the training routine for a given model, train and validation loaders, optimizer, learning rate scheduler, loss function, hooks and given device. It will handle loading of data on device, computing model output, computing loss and applying backward and optimizer step as well as validation.

The training routine is fixed and hooks are called at different steps (`on_batch_begin,  before_backward, on_epoch_end`...) to allow for flexibility and training in various set ups.

Therefore the only thing that mainly needs to be changed are to create custom hooks for a given use case. 

### Hooks

The core hooks include:
- `Validation`: handles keeping track of loss and accuracy during validation (should be generalized to any type of metric)
- `Logging`: handles sending values of collected variables to tensorboard
- `Collect`: general hook to keep track of a given variable. Subcalss of this hook include LRCollect, LossCollect and AccCollect.

Other hooks have been implemented:
- `Knowledge distillation`: For image classification tasks
- `MEAL V2`: For image classification tasks
- `SWA`: Stochastic Weight Averaging
- `VAT`: Virtual Adversarial Training, semi supervised learning technique (needs fix)

### Datasets

An `BaseImageDataset` class is implemented along with two subclasses CIFAR10 and CIFAR100.


### Optim

Contains functionalities to build optimizer and LR scheduler.


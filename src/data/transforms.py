from torchvision import transforms


transforms_map = {
    "HorizontalFlip": transforms.RandomHorizontalFlip,
    "VerticalFlip": transforms.RandomVerticalFlip,
    "Rotation": transforms.RandomRotation,
    "Resize": transforms.Resize,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "Perspective": transforms.RandomPerspective,
    "ColorJitter": transforms.ColorJitter
}

def build_transforms(transforms_list, config):
    img_transforms = []
    for (transf, transf_args) in transforms_list:
        if transf_args is not None:
            img_transforms.append(transforms_map[transf](**transf_args))
        else:
            img_transforms.append(transforms_map[transf]())
    img_transforms.append(transforms.ToTensor())
    if config.MODEL.PRETRAINED:
        img_transforms.append(transforms.Normalize(
            mean =[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]))
    else: # from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        img_transforms.append(transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]))
    return transforms.Compose(img_transforms)
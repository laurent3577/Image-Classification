import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from .utils import acc

def parse_args():
    parser = argparse.ArgumentParser(description='Test Classification Model')

    parser.add_argument('--save-path',
                        help='Save path to model',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args

def main():
	args = parse_args()
	config = torch.load(args.save_path)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = build_model(config)
	model = model.to(device)
	transforms = build_transforms([("Resize", {"size": config.DATASET.INPUT_SIZE})], config)

	dataset = build_dataset(config, split='test', transform=transforms)
	sampler = SequentialSampler(dataset)
	loader = DataLoader(dataset, batch_size=32, sampler=sampler)

	model.eval()
	accuracy = 0
	total = 0

	for img, target in loader:
		outputs = model(img)
		accuracy += acc(outputs, target)
		total += outputs.size(0)

	print("Test Results\n------------\nAccuracy: {.2f} ({}/{})".format(accuracy/len(loader)*100, int(accuracy/len(loader)*total), total))


if __name__ == '__main__':
    main()
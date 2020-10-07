import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from src import build_model, build_dataset, acc

def parse_args():
    parser = argparse.ArgumentParser(description='Test Classification Model')

    parser.add_argument('--save-path',
                        help='Save path to model',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args

def main():
	print("Testing...")
	args = parse_args()
	model = load_from_path(args.save_path)
	transforms = [("Resize", {"size": config.DATASET.INPUT_SIZE})]

	dataset = build_dataset(config, split='test', transform=transforms)
	sampler = SequentialSampler(dataset)
	loader = DataLoader(dataset, batch_size=32, sampler=sampler)

	model.eval()
	accuracy = 0
	total = 0
	with torch.no_grad():
		for img, target in tqdm(loader):
			img = img.to(device)
			target = target.to(device)
			outputs = model(img)
			accuracy += acc(outputs, target)
			total += outputs.size(0)

	print("Test Results\n------------\nAccuracy: {0:.2f} ({1}/{2})".format(accuracy/len(loader)*100, int(accuracy/len(loader)*total), total))


if __name__ == '__main__':
    main()
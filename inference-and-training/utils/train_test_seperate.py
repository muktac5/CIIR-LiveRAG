import json
import glob
import os
import random
import argparse

parser = argparse.ArgumentParser(description="Separate train and test datasets")

parser.add_argument("--datamorgana_data_addr", type=str, help="Path to the directory containing datamorgana data files")
parser.add_argument("--output_dir", type=str, help="Path to the directory where the output files will be saved")

if __name__ == "__main__":
    args = parser.parse_args()
    datamorgana_data_addr_pattern = os.path.join(args.datamorgana_data_addr, "*.json")
    output_dir = args.output_dir
    num_test = 1000

    dataset = []
    for file in glob.glob(datamorgana_data_addr_pattern):
        with open(file, "r") as f:
            data = json.load(f)
            dataset.extend(data)

    random.shuffle(dataset)

    test_dataset = dataset[:num_test]
    train_dataset = dataset[num_test:]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_dataset, f, indent=4)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_dataset, f, indent=4)
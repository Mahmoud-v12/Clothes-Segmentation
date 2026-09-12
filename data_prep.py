from datasets import load_dataset

import config


def load_raw_dataset():
    dataset = load_dataset(config.DATASET_NAME)
    return dataset["train"]


def split_dataset(full_dataset, val_ratio=None, test_ratio=None, seed=None):

    val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
    test_ratio = config.TEST_RATIO if test_ratio is None else test_ratio
    seed = config.SEED if seed is None else seed

    temp_ratio = val_ratio + test_ratio

    split_1 = full_dataset.train_test_split(test_size=temp_ratio, seed=seed)
    train_data = split_1["train"]
    temp_data = split_1["test"]

    split_2 = temp_data.train_test_split(test_size=test_ratio / temp_ratio, seed=seed)
    val_data = split_2["train"]
    test_data = split_2["test"]

    return train_data, val_data, test_data


def load_and_split_dataset():
    full_dataset = load_raw_dataset()
    return split_dataset(full_dataset)


if __name__ == "__main__":
    train_data, val_data, test_data = load_and_split_dataset()
    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

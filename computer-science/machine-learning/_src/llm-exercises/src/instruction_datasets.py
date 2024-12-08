from datasets import dataset_dict, load_dataset


def load_ichikara_instruction_003_001_1() -> dataset_dict.Dataset:
    dataset = load_dataset(
        "json",
        data_files="data/Distribution20241221_all_preprocessed/ichikara-instruction-003-001-1.json",
    )

    dataset["train"] = dataset["train"].rename_column("text", "input")
    return dataset["train"]


def load_ichikara_instruction_all() -> dataset_dict.Dataset:
    dataset = load_dataset(
        "json",
        data_files="data/Distribution20241221_all_preprocessed/*.json",  # avoid README.md
    )
    dataset["train"] = dataset["train"].rename_column("text", "input")
    return dataset["train"]


def load_elyza_tasks_100() -> dataset_dict.Dataset:
    dataset = load_dataset("elyza/ELYZA-tasks-100")
    return dataset["test"]


def load_elyza_tasks_100_TV() -> dataset_dict.Dataset:
    dataset = load_dataset(
        "json",
        data_files="data/elyza-tasks-100-TV_0.jsonl",
    )
    return dataset["train"]


INSTRUCTION_DATASETS = {
    "ichikara-instruction-003-001-1": load_ichikara_instruction_003_001_1,
    "ichikara-instruction-all": load_ichikara_instruction_all,
    "elyza/ELYZA-tasks-100": load_elyza_tasks_100,
    "elyza-tasks-100-TV_0": load_elyza_tasks_100_TV,
}


if __name__ == "__main__":
    for dataset_name, load_func in INSTRUCTION_DATASETS.items():
        dataset = load_func()
        print(dataset)

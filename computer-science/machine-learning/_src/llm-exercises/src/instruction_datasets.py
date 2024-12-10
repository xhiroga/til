from datasets import dataset_dict, load_dataset


def load_ichikara_instruction_all() -> dataset_dict.Dataset:
    dataset = load_dataset("xhiroga/ichikara-instruction-003", "all")
    dataset["train"]["input"] = dataset["train"]["text"]
    return dataset["train"]


def load_elyza_tasks_100() -> dataset_dict.Dataset:
    dataset = load_dataset("elyza/ELYZA-tasks-100")
    return dataset["test"]


def load_elyza_tasks_100_TV() -> dataset_dict.Dataset:
    dataset = load_dataset("xhiroga/ELYZA-tasks-100")
    return dataset["train"]


INSTRUCTION_DATASETS = {
    "ichikara-instruction-all": load_ichikara_instruction_all,
    "elyza/ELYZA-tasks-100": load_elyza_tasks_100,
    "elyza-tasks-100-TV_0": load_elyza_tasks_100_TV,
}


if __name__ == "__main__":
    for dataset_name, load_func in INSTRUCTION_DATASETS.items():
        dataset = load_func()
        print(dataset)

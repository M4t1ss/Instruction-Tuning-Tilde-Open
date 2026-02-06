from torch.utils.data import Dataset
import json
from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm
import random

def format_to_chatml(row, system_message="Tu esi izpalīdzīga mākslīgā intelekta asistente."):

    if 'instruction' in row:
        user_content = row['instruction']
        if 'input' in row and row['input'] != "nan":
            user_content = f"{user_content}\n\nKonteksts:\n{row['input']}"
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row['output']}
        ]
    else:
        return row

def parse_hf_dataset(example, dataset_name):
    """
    Convert different HF dataset schemas into:
    {instruction, input, output}
    """
    sys_msg = [{"role": "system", "content": "Tu esi izpalīdzīga mākslīgā intelekta asistente."}]

    if dataset_name in ["zhengr/ultrachat_200k", "martinsu/latvian-wikipedia-qa-gemma3"]:
        return sys_msg + example["messages"]

    elif dataset_name == "utter-project/EuroBlocks-SFT-Synthetic-1124":

        # Mapping dictionary for the values
        value_map = {"human": "user", "gpt": "assistant"}

        # List comprehension to rebuild the dictionaries
        updated_messages = [
            {
                "role": value_map.get(m["from"], m["from"]), 
                "content": m["value"]
            } 
            for m in example["conversations"]
        ]

        return sys_msg + updated_messages

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def load_and_split_data(
        train_split=0.85,
        test_split=0.1,
        test_size=3000,
        val_size=5000,
        json_paths=None,
        hf_datasets=None
    ):
    """Load and split data into train, validation and test sets."""
    data = []
    if json_paths:
        for path in json_paths:
            with open(path, "r") as f:
                data.extend(json.load(f))

    if hf_datasets:
        for ds_name in hf_datasets:
            splits = get_dataset_split_names(ds_name)
            if "train" in splits:
                fds = load_dataset(ds_name, split="train")
                ds = fds.shuffle(seed=347155).select(range(100000))
            elif "train_sft" in splits:
                fds = load_dataset(ds_name, split="train_sft")
                ds = fds.shuffle(seed=347155).select(range(70000))
            elif "test_sft" in splits:
                ds = load_dataset(ds_name, split="test_sft")
            for ex in tqdm(ds, desc=f"Converting {ds_name}", unit="samples"):
                data.append(parse_hf_dataset(ex, ds_name))
    
    random.shuffle(data)
    N = len(data)
    print(f'Total data: {"{:,}".format(N)}')

    test_portion = test_size
    val_portion = val_size
    train_portion = N - test_portion - val_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Training set length: {"{:,}".format(len(train_data))}")
    print(f"Validation set length: {"{:,}".format(len(val_data))}")
    print(f"Test set length: {"{:,}".format(len(test_data))}")

    return train_data, val_data, test_data

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in tqdm(data, desc="Tokenizing samples", unit="samples"):
            messages = format_to_chatml(entry)

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # encode fully formatted data point
            self.encoded_texts.append(
                tokenizer.encode(prompt)
            )

    def __getitem__(self, ind):
        return self.encoded_texts[ind]

    def __len__(self):
        return len(self.data)
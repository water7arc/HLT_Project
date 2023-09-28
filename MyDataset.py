from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        context = self.data[index]["context"]
        # answer_start = self.data[index]["answers"]["answer_start"][0]
        # answer_end = answer_start + len(self.data[index]["answers"]["text"][0])
        new_context = self.data[index]["answers"]["text"][0] + "</s>" +  context 

        tokenized_in = self.tokenizer(
            new_context,
            text_target=self.data[index]["question"],
            max_length=256,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_in["labels"][tokenized_in["labels"] == self.tokenizer.pad_token_id] = -100
        return {"input_ids": tokenized_in["input_ids"][0], "labels": tokenized_in["labels"][0],
                "attention_mask": tokenized_in["attention_mask"][0]}

    def __len__(self):
        return len(self.data)


class MyDataset_answer(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        context = self.data[index]["context"]
        # answer_start = self.data[index]["answers"]["answer_start"][0]
        # answer_end = answer_start + len(self.data[index]["answers"]["text"][0])
        new_context = self.data[index]["answers"]["text"][0] + "</s>" +  context 

        tokenized_in = self.tokenizer(
            new_context,
            text_target=self.data[index]["question"],
            max_length=256,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_in["labels"][tokenized_in["labels"] == self.tokenizer.pad_token_id] = -100
        return {"input_ids": tokenized_in["input_ids"][0], "labels": tokenized_in["labels"][0],
                "attention_mask": tokenized_in["attention_mask"][0]}

    def __len__(self):
        return len(self.data)
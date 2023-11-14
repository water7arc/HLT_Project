from torch.utils.data import Dataset
import torch


class MyDataset_question_generation(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        contexts = self.data[index]["context"]
        answers = self.data[index]["answers"]
        new_contexts = [answer["text"][0] + "</s>" +  context for context, answer in zip(contexts, answers)] 

        tokenized_in = self.tokenizer(
            new_contexts,
            text_target=self.data[index]["question"],
            max_length=256,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_in["labels"][tokenized_in["labels"] == self.tokenizer.pad_token_id] = -100
        if type(index) == int:
            return {"input_ids": tokenized_in["input_ids"][0], "labels": tokenized_in["labels"][0],
                    "attention_mask": tokenized_in["attention_mask"][0]}
        else:
            return {"input_ids": tokenized_in["input_ids"], "labels": tokenized_in["labels"],
                    "attention_mask": tokenized_in["attention_mask"]}

    def __len__(self):
        return len(self.data)


class MyDataset_answer_generation(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        contexts = self.data[index]["context"]
        answers = [a["text"][0] for a in self.data[index]["answers"]]

        tokenized_in = self.tokenizer(
            contexts,
            text_target=answers,
            max_length=256,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_in["labels"][tokenized_in["labels"] == self.tokenizer.pad_token_id] = -100

        if type(index) == int:
            return {"input_ids": tokenized_in["input_ids"][0], "labels": tokenized_in["labels"][0],
                    "attention_mask": tokenized_in["attention_mask"][0]}
        else:
            return {"input_ids": tokenized_in["input_ids"], "labels": tokenized_in["labels"],
                    "attention_mask": tokenized_in["attention_mask"]}

    def __len__(self):
        return len(self.data)


class MyDataset_answer_generation_qa(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        context = self.data[index]["context"]
        answer = self.data[index]["answers"]

        tokenized_in = self.tokenizer(
            context,
            max_length=256,
            truncation="only_first",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offset = tokenized_in.pop("offset_mapping")[0]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])

        context_end = torch.count_nonzero(tokenized_in["attention_mask"]).item() - 2

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[0, 0] > end_char or offset[context_end, 1] < start_char:
            # print(offset[0, 0])
            # print(offset[context_end, 1])
            start_position = 0
            end_position = 0
        else:
            # Otherwise it's the start and end token positions
            idx = 0
            while idx <= context_end and offset[idx, 0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= 0 and offset[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

        tokenized_in["start_positions"] = start_position
        tokenized_in["end_positions"] = end_position
        tokenized_in["input_ids"] = tokenized_in["input_ids"][0]
        tokenized_in["attention_mask"] = tokenized_in["attention_mask"][0]
        tokenized_in.pop("token_type_ids")
        return tokenized_in

    def __len__(self):
        return len(self.data)  


class MyDataset_question_answering(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        context = self.data[index]["context"]
        # answer_start = self.data[index]["answers"]["answer_start"][0]
        # answer_end = answer_start + len(self.data[index]["answers"]["text"][0])

        tokenized_in = self.tokenizer(
            context,
            text_target=self.data[index]["answers"]["text"][0],
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


class MyDataset_e2e_question_generation(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        contexts = self.data[index]["context"]
        answers = self.data[index]["answers"]

        questions = self.data[index]["question"]
        question_anwers = [q + "</s>" + a for (q, a) in zip(questions, answers)]

        tokenized_in = self.tokenizer(
            contexts,
            text_target=question_anwers,
            max_length=256,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_in["labels"][tokenized_in["labels"] == self.tokenizer.pad_token_id] = -100

        if type(index) == int:
            return {"input_ids": tokenized_in["input_ids"][0], "labels": tokenized_in["labels"][0],
                    "attention_mask": tokenized_in["attention_mask"][0]}
        else:
            return {"input_ids": tokenized_in["input_ids"], "labels": tokenized_in["labels"],
                    "attention_mask": tokenized_in["attention_mask"]}

    def __len__(self):
        return len(self.data)

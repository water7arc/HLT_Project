from torch.utils.data import Dataset
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return self.tokenizer(self.data[index]["context"], return_tensors="pt").input_ids.to("cuda"), \
               self.tokenizer(self.data[index]["answers"]["text"][0], return_tensors="pt").input_ids.to("cuda"), \
               self.tokenizer(self.data[index]["question"], return_tensors="pt").input_ids.to("cuda")

    def __len__(self):
        return len(self.data)
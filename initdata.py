from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
import hashlib

dist = Path(__file__).parent 
data = dist / "data" / "pretrain_t2t_mini.jsonl" 
tokenizer = Tokenizer.from_file(str(dist / "tokenizer.json"))
start_id = tokenizer.token_to_id("<START>")
end_id = tokenizer.token_to_id("<END>")
empty_id = tokenizer.token_to_id("<EMPTY>")
length = 256 
output = dist / "data" / "cache" 
output.mkdir(parents = True, exist_ok = True)

dataset = load_dataset("json", data_files = str(data), split = "train")
print(dataset)
def check_empty(text):
    return bool(text["text"].strip()) 
dataset = dataset.filter(check_empty)
def tokenize_batch(batch):
    texts = batch["text"]
    encoding = tokenizer.encode_batch(texts, add_special_tokens = False)
    token_ids = []
    split_keys = []
    for text, tokens in zip(texts, encoding):
        token_ids.append([start_id] + tokens.ids + [end_id])
        split_keys.append(hashlib.sha256(text.encode("utf-8")).digest()[0])
    return {"token_ids" : token_ids, "split_keys": split_keys}
def perform(data, fun):
    return data.map(fun, batched = True, batch_size = 1000, remove_columns = data.column_names)
dataset = perform(dataset, tokenize_batch)
dataset = dataset.shuffle()
def is_train(x):
    return x["split_keys"] >= 25
def is_val(x):
    return x["split_keys"] < 25
train_data = dataset.filter(is_train)
val_data = dataset.filter(is_val)
print("训练文本数量：", len(train_data))
print("验证文本数量：", len(val_data))
def cut_batch(batch):
    res = []
    for ids in batch["token_ids"]:
        for i in range(0, len(ids) - 1, length):
            x = ids[i : i + length + 1]
            res.append(x + [empty_id] * (length + 1 - len(x)))
    return {"token_ids" : res}
train_data = perform(train_data, cut_batch)
val_data = perform(val_data, cut_batch)
train_data.save_to_disk(str(output / "train"))
val_data.save_to_disk(str(output / "val"))
print("训练片段数量：", len(train_data))
print("验证片段数量：", len(val_data))
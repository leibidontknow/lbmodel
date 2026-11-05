from pathlib import Path
from datasets import load_dataset 
from tokenizers import Tokenizer 
from datasets import load_dataset, Features, Sequence, Value
dist = Path(__file__).parent 
tokenizer = Tokenizer.from_file(str(dist / "tokenizer.json"));
data = load_dataset(
    "json",
    data_files={
        "train": str(
            dist / "data" / "sft_zh_million_v1_train.jsonl"
        ),
        "val": str(
            dist / "data" / "sft_zh_million_v1_val.jsonl"
        ),
    },
)
print(data)
start_id = tokenizer.token_to_id("<START>")
end_id = tokenizer.token_to_id("<END>")
empty_id = tokenizer.token_to_id("<EMPTY>")
length = 768
def encode(text):
    token_ids = []
    labels = []
    for msg in text["conversations"]:
        role = msg["role"]
        content = msg["content"]
        prefix = [start_id] + tokenizer.encode(
            role + "\n", add_special_tokens=False
        ).ids
        body = tokenizer.encode(
            content, add_special_tokens=False
        ).ids + [end_id]
        token_ids += prefix + body
        labels += [-100] * len(prefix)
        labels += body if role == "assistant" else [-100] * len(body)
    token_ids = token_ids[:length + 1]
    labels = labels[:length + 1]
    pad = length + 1 - len(token_ids)
    return {
        "token_ids": token_ids[:length + 1],
        "labels": labels[:length + 1]
    }
data = data.map(encode, remove_columns=data["train"].column_names, features=Features({
    "token_ids": Sequence(Value("int16")), 
    "labels": Sequence(Value("int16"))
}))
data = data.filter(lambda x: any(y != -100 for y in x["labels"][1:]))
output = dist / "data" / "million_sft_cache"
output.mkdir(parents=True, exist_ok=True)
data["train"].save_to_disk(str(output / "train"))
data["val"].save_to_disk(str(output / "val"))
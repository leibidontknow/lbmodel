from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers 
import json
import hashlib
from pathlib import Path

tokenizer = Tokenizer(models.BPE()) 
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) 
tokenizer.decoder = decoders.ByteLevel() 
trainer = trainers.BpeTrainer(
    vocab_size = 8192, 
    min_frequency = 2, 
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(), 
    special_tokens = ["<START>", "<END>", "<EMPTY>"]
)

data = Path(__file__).parent / "data" / "pretrain_t2t_mini.jsonl"
def get_text():
    tot = 0
    with data.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            s = json.loads(line)["text"]
            if not s.strip():
                continue
            hsh = hashlib.sha256(s.encode("utf-8")).digest()
            if(hsh[0] < 25):
                continue
            yield s
            tot += len(s)
            if tot >= 20000000:
                break
tokenizer.train_from_iterator(get_text(), trainer = trainer)
tokenizer.save(str(Path(__file__).with_name("tokenizer.json")))
print("valcsiz: ", tokenizer.get_vocab_size())
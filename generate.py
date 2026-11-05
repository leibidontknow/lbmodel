from pathlib import Path 
import torch 
from tokenizers import Tokenizer 
from model import Model 
from tokenizers.decoders import DecodeStream 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
chkpoint = torch.load(
    Path(__file__).with_name("million_sft_weights.pt"), 
    map_location = "cpu", 
    weights_only = True
)
model = Model(**chkpoint["config"]).to(device)
model.load_state_dict(chkpoint["model"])
del chkpoint
model.eval()
tokenizer = Tokenizer.from_file(str(Path(__file__).with_name("tokenizer.json")))
start_id = tokenizer.token_to_id("<START>")
end_id = tokenizer.token_to_id("<END>")
empty_id = tokenizer.token_to_id("<EMPTY>")
temperature = 0.3
text = input("输入：")
ids = (
    [start_id]
    + tokenizer.encode("user\n", add_special_tokens=False).ids
    + tokenizer.encode(text, add_special_tokens=False).ids
    + [end_id]
    + [start_id]
    + tokenizer.encode("assistant\n", add_special_tokens=False).ids
)
prompt_len = len(ids) 
kv_cache = None
if prompt_len > model.max_len:
    raise ValueError("文本过长")
stream = DecodeStream(skip_special_tokens=True)
print("回答：", end="", flush=True)
with torch.no_grad():
    for i in range(model.max_len - prompt_len):
        if kv_cache is None:
            cur_ids = ids 
        else: 
            cur_ids = ids[-1:]
        x = torch.tensor(
            [cur_ids],
            dtype = torch.long, 
            device = device 
        )
        logits, kv_cache = model(x, kv_cache, True)
        logits = logits[0, -1, :]
        logits[start_id] = float("-inf")
        logits[empty_id] = float("-inf")
        seen = set(ids[prompt_len:])
        for token_id in seen:
            if logits[token_id] > 0:
                logits[token_id] /= 1.1
            else:
                logits[token_id] *= 1.1
        values, indices = torch.topk(logits, k=40)
        probs = torch.softmax(values / temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1).item()
        nxt_tok = indices[choice].item()
        if nxt_tok == end_id:
            break
        ids.append(nxt_tok)
        piece = stream.step(tokenizer, nxt_tok)
        if piece is not None:
            print(piece, end="", flush=True)
print()
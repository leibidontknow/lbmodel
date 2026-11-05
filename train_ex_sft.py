from pathlib import Path 
import torch
from torch.nn import functional as F    
from torch.nn.utils.rnn import pad_sequence 
from tokenizers import Tokenizer
from model import Model 
import sys 
import math 
import csv 
from datasets import load_from_disk 

tokenizer = Tokenizer.from_file(str(Path(__file__).parent / "tokenizer.json"))
start_id = tokenizer.token_to_id("<START>")
end_id = tokenizer.token_to_id("<END>")
empty_id = tokenizer.token_to_id("<EMPTY>")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
use_bf16 = False 
if device == "cuda":
    use_bf16 = torch.cuda.is_bf16_supported() 
print("device:", device, "BF16混合精度:", use_bf16)
checkpoint = torch.load(
    Path(__file__).with_name("ex_sft_weights.pt"), 
    map_location="cpu",
    weights_only=True 
)
config = checkpoint["config"]
config["max_len"] = 32768
model = Model(**config).to(device)
model.load_state_dict(checkpoint["model"])
del checkpoint 
lr = 1e-5 
optimizer = torch.optim.AdamW(model.parameters(), lr)
start_step = 0
batch_siz = 32
ignore_id = -100
cache_dir = Path(__file__).parent / "data" / "million_sft_cache"
data_sets = {
    "train" : load_from_disk(str(cache_dir / "train")), 
    "chk": load_from_disk(str(cache_dir / "val"))
}
def get_batch(split, b_siz, start_b = 0):
    data = data_sets[split]
    start_idx = start_b * b_siz 
    for i in range(start_idx, len(data), b_siz):
        batch = data[i:i + b_siz]
        res = pad_sequence(
            [torch.as_tensor(x, dtype=torch.long) for x in batch["token_ids"]], 
            batch_first=True, 
            padding_value=empty_id 
        )
        mx_len = (res != empty_id).sum(dim=1).max().item()
        res = res[:, :mx_len].to(device)
        labels = pad_sequence(
            [torch.as_tensor(x, dtype=torch.long) for x in batch["labels"]], 
            batch_first=True, 
            padding_value=-100
        )
        labels = labels[:, :mx_len].to(device)
        yield res[:, :-1], labels[:, 1:]
def eval_loss(b_cnt):
    model.eval()
    loss = torch.tensor(0.0, device=device)
    tok_cnt = torch.tensor(0, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (b_in, b_tar) in enumerate(get_batch("chk", batch_siz), start = 1):
            with torch.autocast(device, torch.bfloat16, use_bf16):
                logits = model(b_in)
                b_loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer.get_vocab_size()),
                    b_tar.reshape(-1), 
                    ignore_index = ignore_id, 
                    reduction = "sum"
                )
            loss += b_loss 
            tok_cnt = tok_cnt + (b_tar != ignore_id).sum()
            if i >= b_cnt:
                break 
    model.train()
    if tok_cnt.item() == 0:
        raise ValueError("token all gunmu")
    return (loss / tok_cnt).item()
def save_chkpoint(step):
    torch.save(
        {
            "config": config, 
            "model": model.state_dict()
        }, 
        Path(__file__).with_name("million_sft_weights.pt"), 
    )
model.train()
step = start_step
loss_path = Path(__file__).with_name("million_sft_loss.csv")
loss_path.write_text("step,type,loss\n", encoding="utf-8")
def record_loss(kind, value):
    with loss_path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([step, kind, value])
batch_cnt = (len(data_sets["train"]) + batch_siz - 1) // batch_siz
total_steps = batch_cnt
warmup_steps = 200
min_lr = 0.1 * lr 
start_b = start_step
for b_in, b_tar in get_batch("train", batch_siz, start_b):
    optimizer.zero_grad()
    with torch.autocast(device, torch.bfloat16, use_bf16):
        logits = model(b_in) 
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.get_vocab_size()), 
            b_tar.reshape(-1), 
            ignore_index = ignore_id
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
    t = step + 1
    if t <= warmup_steps:
        ratio = t / warmup_steps
    else:
        progress = min(
            (t - warmup_steps) / (total_steps - warmup_steps),
            1.0,
        )
        ratio = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
    optimizer.param_groups[0]["lr"] = lr * ratio
    optimizer.step()
    step += 1        
    if step % 1000 == 0:
        val_loss = eval_loss(b_cnt=20)
        print(f"step:{step} loss:{val_loss}")
        save_chkpoint(step)
        record_loss("val", val_loss)
    if step == start_step + 1 or step % 100 == 0 or step == total_steps:
        print(
            f"进度：{step}/{total_steps} ({step / total_steps:.2%}) "
            f"loss={loss.item():.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        record_loss("train", loss.item())
save_chkpoint(step)

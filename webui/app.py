from pathlib import Path

import gradio as gr
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from model import Model


ROOT = Path(__file__).parent
WEIGHTS_FILE = "million_sft_weights.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torch.load(
    ROOT / WEIGHTS_FILE,
    map_location="cpu",
    weights_only=True,
)

config = dict(weights["config"])
config["max_len"] = 32768

model = Model(**config)
model.load_state_dict(weights["model"])
del weights

model = model.to(device).eval()

tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
start_id = tokenizer.token_to_id("<START>")
end_id = tokenizer.token_to_id("<END>")
empty_id = tokenizer.token_to_id("<EMPTY>")

MAX_INPUT_TOKENS = model.max_len - 1

assistant_prefix = [start_id] + tokenizer.encode(
    "assistant\n",
    add_special_tokens=False,
).ids
def encode_message(role, content):
    return (
        [start_id]
        + tokenizer.encode(
            role + "\n",
            add_special_tokens=False,
        ).ids
        + tokenizer.encode(
            content,
            add_special_tokens=False,
        ).ids
        + [end_id]
    )
def make_prompt(message, history, use_context=False):
    current = encode_message("user", message) + assistant_prefix
    if len(current) > MAX_INPUT_TOKENS:
        raise gr.Error("这条消息太长，请缩短后重试。")
    if not use_context:
        return current
    turns = []
    pending_user = None
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            content = "".join(
                block["text"]
                for block in content
                if block.get("type") == "text"
            )

        if not isinstance(content, str):
            continue
        if role == "user":
            pending_user = encode_message("user", content)
        elif role == "assistant" and pending_user is not None:
            turns.append(
                pending_user + encode_message("assistant", content)
            )
            pending_user = None
    selected = []
    total = len(current)
    for turn in reversed(turns):
        if total + len(turn) > MAX_INPUT_TOKENS:
            break
        selected.append(turn)
        total += len(turn)

    ids = []
    for turn in reversed(selected):
        ids.extend(turn)
    ids.extend(current)
    return ids
def chat(message, history, temperature, repetition_penalty=1.1, use_context=False):
    ids = make_prompt(message, history, use_context)
    kv_cache = None
    generated = set()
    stream = DecodeStream(skip_special_tokens=True)
    answer = ""
    for _ in range(model.max_len - len(ids)):
        cur_ids = ids if kv_cache is None else ids[-1:]
        x = torch.tensor(
            [cur_ids],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            logits, kv_cache = model(x, kv_cache, True)
            logits = logits[0, -1, :].float().clone()
            logits[start_id] = float("-inf")
            logits[empty_id] = float("-inf")
            if repetition_penalty != 1.0 and generated:
                seen = list(generated)
                scores = logits[seen]
                logits[seen] = torch.where(
                    scores > 0,
                    scores / repetition_penalty,
                    scores * repetition_penalty,
                )
            values, indices = torch.topk(
                logits,
                k=min(40, logits.numel()),
            )
            probs = torch.softmax(values / temperature, dim=-1)
            choice = torch.multinomial(probs, 1).item()
            token_id = indices[choice].item()
        del logits, x
        if token_id == end_id:
            break
        ids.append(token_id)
        generated.add(token_id)

        piece = stream.step(tokenizer, token_id)
        if piece:
            answer += piece
            yield answer
    if not answer:
        yield ""
def respond(message, history, temperature, repetition_penalty, use_context):
    if not message.strip():
        yield "", history
        return

    history = list(history or [])
    display = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield "", display
    for answer in chat(message, history, temperature, repetition_penalty, use_context):
        yield "", display[:-1] + [
            {"role": "assistant", "content": answer}
        ]


with gr.Blocks() as demo:
    gr.Markdown("# lbmodel\n由 lbdontknow 创建的小语言模型")
    chatbot = gr.Chatbot()
    with gr.Row():
        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.3,
            step=0.05,
            label="Temperature",
            info="调低更集中，调高更多样",
        )
        repetition_penalty = gr.Slider(
            minimum=1.0,
            maximum=1.5,
            value=1.1,
            step=0.05,
            label="重复惩罚",
            info="1.0 为关闭；可尝试 1.1 抑制重复。",
        )
        use_context = gr.Checkbox(
            value=False,
            label="启用上下文",
            info="勾选后携带之前的对话；不勾选时只回答当前消息。",
        )
    message = gr.Textbox(label="消息", placeholder="输入消息后按回车发送")
    with gr.Row():
        send = gr.Button("发送", variant="primary")
        stop = gr.Button("停止生成")
        clear = gr.Button("清空上下文")

    event_options = dict(
        fn=respond,
        inputs=[message, chatbot, temperature, repetition_penalty, use_context],
        outputs=[message, chatbot],
        concurrency_limit=1,
        concurrency_id="model_generation",
    )
    send_event = send.click(**event_options)
    submit_event = message.submit(**event_options)
    stop.click(fn=None, cancels=[send_event, submit_event])
    clear.click(
        fn=lambda: ("", []),
        outputs=[message, chatbot],
        cancels=[send_event, submit_event],
        queue=False,
    )
if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )

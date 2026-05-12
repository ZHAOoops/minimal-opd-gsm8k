import argparse, re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_gt(ans):
    return ans.split("####")[-1].strip().replace(",", "")

def extract_pred(text):
    # Prefer the first complete "#### number".
    # This avoids repeated outputs like "#### 3#### 3#### ..." breaking parsing.
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Fallback: use the last number in the response.
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not nums:
        return None
    return nums[-1].replace(",", "")

def same_number(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except:
        return False


def get_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)
    eos_ids = list(dict.fromkeys(eos_ids))
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids

def make_plain_prompt(q):
    return (
        "Solve the following math problem step by step. "
        "Put the final answer after ####.\n\n"
        f"Question: {q}\n"
        "Answer:"
    )

def make_chat_prompt(tokenizer, q):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Solve problems step by step."
        },
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "Put the final answer after ####.\n\n"
                f"Question: {q}"
            )
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = get_eos_ids(tokenizer)
    print("eos_ids:", eos_ids)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    ds = load_dataset("gsm8k", "main")["test"]
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct, total = 0, 0

    for start in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[start:start + args.batch_size]
        qs = batch["question"]
        gts = batch["answer"]

        if args.chat_template:
            prompts = [make_chat_prompt(tokenizer, q) for q in qs]
        else:
            prompts = [make_plain_prompt(q) for q in qs]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

        gen = outputs[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for q, gt_raw, pred_text in zip(qs, gts, texts):
            gt = extract_gt(gt_raw)
            pred = extract_pred(pred_text)
            ok = pred is not None and same_number(pred, gt)

            correct += int(ok)
            total += 1

            if total <= 3:
                print("\n" + "=" * 80)
                print("Q:", q)
                print("GT:", gt)
                print("PRED:", pred)
                print("OK:", ok)
                print("GEN:", pred_text[:1000])

    print("\nRESULT")
    print("model:", args.model)
    print("chat_template:", args.chat_template)
    print("correct:", correct)
    print("total:", total)
    print("acc:", correct / total if total else 0)

if __name__ == "__main__":
    main()

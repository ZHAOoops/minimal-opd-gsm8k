import os
import time
import random
import argparse
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_chat_prompt(tokenizer, question):
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
                f"Question: {question}"
            )
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)

    eos_ids = list(dict.fromkeys(eos_ids))
    if len(eos_ids) == 1:
        return eos_ids[0]
    return eos_ids


def truncate_after_final_answer(full_ids, prompt_len, tokenizer, pattern, stop_token_id=None):
    """
    Batch size = 1 version.
    Keep the sampled tokens up to the first occurrence of something like:
      #### 72
      #### -3.5

    Important: this does NOT use ground truth. It only looks at student's own output.
    """
    completion_ids = full_ids[0, prompt_len:].detach().cpu().tolist()
    if len(completion_ids) == 0:
        return full_ids, False

    text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    m = re.search(pattern, text)
    if m is None:
        return full_ids, False

    target_chars = m.end()
    cut_n = len(completion_ids)

    # Find the smallest token prefix whose decoded text covers the matched answer.
    # This preserves sampled token ids instead of re-tokenizing.
    for i in range(1, len(completion_ids) + 1):
        prefix_text = tokenizer.decode(completion_ids[:i], skip_special_tokens=False)
        if len(prefix_text) >= target_chars:
            cut_n = i
            break

    new_full_ids = full_ids[:, :prompt_len + cut_n].contiguous()

    # Important: after truncating at "#### number", append <|im_end|>
    # so the KL includes the teacher's stopping distribution.
    if stop_token_id is not None:
        if new_full_ids[0, -1].item() != stop_token_id:
            stop = torch.tensor([[stop_token_id]], dtype=new_full_ids.dtype, device=new_full_ids.device)
            new_full_ids = torch.cat([new_full_ids, stop], dim=1)

    return new_full_ids, True


def exact_kl_loss(student_logits, teacher_logits, prompt_len, direction="rkl"):
    """
    student_logits / teacher_logits: [1, seq_len, vocab]
    prompt_len: prompt token 数量

    completion token 的预测位置是:
      logits[prompt_len - 1] -> predicts first completion token
      ...
      logits[seq_len - 2]    -> predicts last completion token

    所以切片为 logits[:, prompt_len-1:-1, :]
    """
    start = max(prompt_len - 1, 0)

    s = student_logits[:, start:-1, :].float()
    t = teacher_logits[:, start:-1, :].float().detach()

    if s.shape[1] == 0:
        return None, 0

    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)

    if direction == "rkl":
        # D_KL(student || teacher)
        s_prob = s_logp.exp()
        kl = (s_prob * (s_logp - t_logp)).sum(dim=-1)
    elif direction == "fkl":
        # D_KL(teacher || student)
        t_prob = t_logp.exp()
        kl = (t_prob * (t_logp - s_logp)).sum(dim=-1)
    else:
        raise ValueError(f"unknown kl direction: {direction}")

    return kl.mean(), s.shape[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)

    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--truncate-after-answer", action="store_true")
    parser.add_argument("--answer-pattern", type=str, default=r"####\s*[-+]?\d[\d,]*(?:\.\d+)?")
    parser.add_argument("--save-every", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--kl-direction", type=str, default="rkl", choices=["rkl", "fkl"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=50)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_ids = get_eos_ids(tokenizer)
    print("eos_ids:", eos_ids)
    print("pad_token_id:", tokenizer.pad_token_id)

    answer_stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(answer_stop_token_id, int) or answer_stop_token_id < 0:
        answer_stop_token_id = tokenizer.eos_token_id
    print("answer_stop_token_id:", answer_stop_token_id)

    print("Loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("Loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    student.train()

    teacher_vocab = teacher.get_input_embeddings().weight.shape[0]
    student_vocab = student.get_input_embeddings().weight.shape[0]
    print("teacher vocab:", teacher_vocab)
    print("student vocab:", student_vocab)
    assert teacher_vocab == student_vocab, "teacher/student vocab size mismatch"

    try:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=True,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    print("Loading GSM8K train set...")
    dataset = load_dataset("gsm8k", "main")["train"]
    dataset = dataset.shuffle(seed=args.seed)

    if args.train_limit and args.train_limit > 0:
        dataset = dataset.select(range(min(args.train_limit, len(dataset))))

    print("train examples:", len(dataset))
    print("kl direction:", args.kl_direction)
    print("grad_accum_steps:", args.grad_accum_steps)

    global_micro_step = 0
    global_optim_step = 0
    running_loss = []
    running_tokens = []
    running_truncated = []
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    stop_training = False

    for epoch in range(args.num_epochs):
        print(f"\n===== epoch {epoch + 1}/{args.num_epochs} =====")
        epoch_dataset = dataset.shuffle(seed=args.seed + epoch)

        for ex_idx, example in enumerate(epoch_dataset):
            question = example["question"]
            prompt = build_chat_prompt(tokenizer, question)

            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda")

            prompt_len = prompt_inputs["input_ids"].shape[1]

            if prompt_len >= args.max_seq_len - 1:
                continue

            gen_max_new = min(args.max_new_tokens, args.max_seq_len - prompt_len)
            if gen_max_new <= 0:
                continue

            # 1. on-policy: 当前 student 自己生成 completion
            student.eval()
            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=prompt_inputs["input_ids"],
                    attention_mask=prompt_inputs["attention_mask"],
                    max_new_tokens=gen_max_new,
                    do_sample=(args.temperature > 0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
                if args.temperature > 0:
                    gen_kwargs["temperature"] = args.temperature
                    gen_kwargs["top_p"] = args.top_p

                full_ids = student.generate(**gen_kwargs)
                full_ids = full_ids.detach().clone()

                did_truncate = False
                if args.truncate_after_answer:
                    full_ids, did_truncate = truncate_after_final_answer(
                        full_ids=full_ids,
                        prompt_len=prompt_len,
                        tokenizer=tokenizer,
                        pattern=args.answer_pattern,
                        stop_token_id=answer_stop_token_id,
                    )

            student.train()

            if full_ids.shape[1] > args.max_seq_len:
                full_ids = full_ids[:, :args.max_seq_len]

            full_len = full_ids.shape[1]
            completion_len = full_len - prompt_len
            if completion_len <= 0:
                continue

            attention_mask = torch.ones_like(full_ids, device="cuda")

            # 2. teacher/student 看完全相同的 prompt + student completion
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits

            student_logits = student(
                input_ids=full_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

            # 3. 只在 completion 部分算 exact KL
            loss, loss_tokens = exact_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                prompt_len=prompt_len,
                direction=args.kl_direction,
            )

            if loss is None:
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: bad loss, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            global_micro_step += 1
            running_loss.append(loss.item())
            running_tokens.append(loss_tokens)
            running_truncated.append(1 if did_truncate else 0)

            if global_micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_optim_step += 1

                if global_optim_step % args.log_every == 0:
                    avg_loss = sum(running_loss[-args.log_every:]) / min(len(running_loss), args.log_every)
                    avg_tokens = sum(running_tokens[-args.log_every:]) / min(len(running_tokens), args.log_every)
                    avg_trunc = sum(running_truncated[-args.log_every:]) / min(len(running_truncated), args.log_every)
                    elapsed = time.time() - start_time
                    mem = torch.cuda.max_memory_allocated() / 1024**3

                    print(
                        f"optim_step={global_optim_step} "
                        f"micro_step={global_micro_step} "
                        f"loss={avg_loss:.6f} "
                        f"loss_tokens={avg_tokens:.1f} "
                        f"trunc={avg_trunc:.2f} "
                        f"mem_gb={mem:.2f} "
                        f"elapsed_min={elapsed/60:.1f}"
                    )

                if global_optim_step % args.sample_every == 0:
                    completion_ids = full_ids[0, prompt_len:]
                    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
                    print("\n--- sample completion ---")
                    print(completion[:1000])
                    print("--- end sample ---\n")

                if args.save_every and args.save_every > 0 and global_optim_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{global_optim_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    print("Saving intermediate checkpoint to:", ckpt_dir)
                    student.save_pretrained(ckpt_dir, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt_dir)

                if args.max_train_steps and global_optim_step >= args.max_train_steps:
                    stop_training = True
                    break

        if stop_training:
            break

    # 如果最后还有未 step 的梯度，做一次 optimizer step
    if global_micro_step % args.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_optim_step += 1

    print("\nTraining finished.")
    print("total micro steps:", global_micro_step)
    print("total optimizer steps:", global_optim_step)

    os.makedirs(args.output_dir, exist_ok=True)
    print("Saving student to:", args.output_dir)
    student.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

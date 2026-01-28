import argparse
import glob
import json
import os
import random
import re
import evaluate
import torch
import tqdm

# === 1. 安全导入 vLLM ===
try:
    import vllm
except ImportError:
    vllm = None

from eval.utils import (dynamic_import_function, generate_completions,
                        load_hf_lm_and_tokenizer, query_openai_chat_model)

exact_match = evaluate.load("exact_match")

task_names = [
    "date_understanding",
    "logical_deduction_five_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "object_counting",
    "word_sorting",
    "hyperbaton",
    "sports_understanding",
    "boolean_expressions",
    "tracking_shuffled_objects_seven_objects",
    "ruin_names",
    "tracking_shuffled_objects_three_objects",
    "causal_judgement",
    "reasoning_about_colored_objects",
    "logical_deduction_seven_objects",
    "temporal_sequences",
    "salient_translation_error_detection",
    "tracking_shuffled_objects_five_objects",
    "geometric_shapes",
    "disambiguation_qa",
    "dyck_languages",
    "navigate",
    "formal_fallacies",
    "web_of_lies",
    "snarks",
    "penguins_in_a_table",
    "logical_deduction_three_objects"
]


def main(args):
    random.seed(42)

    all_tasks = {}
    # === 修复 1: 路径修正，直接在 data_dir 下找 json ===
    task_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    
    # 如果没找到，尝试在子目录找（兼容旧逻辑）
    if not task_files:
        task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))

    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            try:
                raw_data = json.load(f)
                
                # === 修复 2: 自适应数据格式 (解决 KeyError: 'examples') ===
                if isinstance(raw_data, dict) and "examples" in raw_data:
                    examples = raw_data["examples"]
                elif isinstance(raw_data, list):
                    examples = raw_data
                elif isinstance(raw_data, dict) and "data" in raw_data: # HuggingFace format
                    examples = raw_data["data"]
                else:
                    # 最后的尝试：假设整个 dict 就是一条数据，或者格式不匹配
                    print(f"[Warning] Could not find list in {task_file}, skipping.")
                    continue

                all_tasks[task_name] = examples

                # === 修复 3: 安全采样 (解决 Sample larger than population) ===
                if args.max_num_examples_per_task:
                    num_to_sample = min(len(all_tasks[task_name]), args.max_num_examples_per_task)
                    all_tasks[task_name] = random.sample(all_tasks[task_name], num_to_sample)
            
            except Exception as e:
                print(f"[Error] Failed to load {task_file}: {e}")
                continue

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    # 如果没找到，尝试向上找一级（兼容性）
    if not cot_prompt_files:
         cot_prompt_files = glob.glob(os.path.join(args.data_dir, "../cot-prompts", "*.txt"))

    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                        assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                        answer = prompt_field.split(
                            "So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)
            all_prompts[task_name] = task_prompt

    # === 修复 4: 交集逻辑 (解决 No common tasks found) ===
    task_keys = set(all_tasks.keys())
    prompt_keys = set(all_prompts.keys())
    common_keys = task_keys.intersection(prompt_keys)
    
    if len(common_keys) < len(task_keys) or len(common_keys) < len(prompt_keys):
        print("\n[Info] Task/Prompt alignment:")
        print(f"  Tasks loaded: {len(task_keys)}")
        print(f"  Prompts loaded: {len(prompt_keys)}")
        print(f"  Common tasks: {len(common_keys)}")

    if len(common_keys) == 0:
        raise ValueError("No common tasks found! Check filenames in data_dir vs cot-prompts.")

    all_tasks = {k: v for k, v in all_tasks.items() if k in common_keys}
    all_prompts = {k: v for k, v in all_prompts.items() if k in common_keys}

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    # Load model
    if args.model_name_or_path:
        if args.use_vllm:
            if vllm is None:
                raise ImportError("vllm not installed.")
            print("Loading vllm model...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                max_num_batched_tokens=4096,
                trust_remote_code=True, # 建议加上
                gpu_memory_utilization=0.9, # 防止 OOM
            )
            tokenizer = model.llm_engine.tokenizer.tokenizer if hasattr(model.llm_engine.tokenizer, 'tokenizer') else model.get_tokenizer()
        else:
            print("Loading model and tokenizer with huggingface...")
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
                convert_to_bf16=args.convert_to_bf16,
                convert_to_half=args.convert_to_half,
            )

        # === 修复 5: Llama-3 / Qwen 万能补丁 (解决无限生成/速度慢) ===
        print(f"[DEBUG] Original Tokenizer EOS: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
        
        # 自动修复 EOS
        if tokenizer.eos_token_id is None:
            # Llama 3
            try:
                test_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
                if isinstance(test_id, int) and test_id != tokenizer.unk_token_id:
                    tokenizer.eos_token = "<|end_of_text|>"
                    tokenizer.eos_token_id = test_id
                    print("  -> Detected Llama-3 style EOS")
            except: pass
            
            # Qwen
            if tokenizer.eos_token_id is None:
                for cand in ["<|im_end|>", "<|endoftext|>"]:
                    try:
                        if cand in tokenizer.get_vocab():
                            tokenizer.eos_token = cand
                            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(cand)
                            print(f"  -> Detected Qwen style EOS: {cand}")
                            break
                    except: pass

        # 自动修复 Pad
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0
            print(f"  -> Set PAD Token ID to {tokenizer.pad_token_id}")

        # 强制同步 Config (HF 生成必须)
        if not args.use_vllm:
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model, "generation_config"):
                model.generation_config.eos_token_id = tokenizer.eos_token_id
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            print(f"[Fixed] Model Config synced. EOS={model.config.eos_token_id}")
        # ==============================================================

    performance = {}

    if args.subtask is not None:
        tasks = [task_names[args.subtask]]
    else:
        tasks = all_tasks.keys()

    # === 修复 6: 增强 Stop Tokens ===
    stop_tokens = ["\n\n", "Question:", "Q:", "<|end_of_text|>", "<|im_end|>", "<|eot_id|>"]

    for task_name in tqdm.tqdm(tasks, desc="Evaluating"):
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]
        if args.model_name_or_path:
            # prepare prompts
            prompts = []
            chat_formatting_function = dynamic_import_function(
                args.chat_formatting_function) if args.use_chat_format else None
            for example in task_examples:
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"]
                if args.use_chat_format:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_formatting_function(messages, add_bos=False)
                    if prompt[-1] in ["\n", " "]:
                        prompt += "A:"
                    else:
                        prompt += " A:"
                else:
                    prompt += "\n\nA:" # Base model standard format
                prompts.append(prompt)
            
            # generate with vllm
            if args.use_vllm:
                sampling_params = vllm.SamplingParams(
                    temperature=0,
                    max_tokens=512,
                    stop=stop_tokens, # 使用增强的 stop tokens
                )
                generations = model.generate(prompts, sampling_params)
                prompt_to_output = {
                    g.prompt: g.outputs[0].text for g in generations
                }
                outputs = [prompt_to_output[prompt]
                           if prompt in prompt_to_output else "" for prompt in prompts]
            # generate with hf model
            else:
                stop_id_sequences = []
                for st in stop_tokens:
                    try:
                        ids = tokenizer.encode(st, add_special_tokens=False)
                        if len(ids) > 0: stop_id_sequences.append(ids)
                    except: pass
                
                # 特别增加 new line 的检测
                nl_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
                stop_id_sequences.append([nl_token, nl_token]) # \n\n

                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=512,
                    do_sample=False,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                    stop_id_sequences=stop_id_sequences
                )
        else:
            # OpenAI logic (unchanged)
            instances = []
            for i, example in enumerate(task_examples):
                prompt = task_prompt.strip() + "\n\nQ: " + \
                    example["input"] + "\nA:"
                instances.append({
                    "id": example["id"] if "id" in example else i,
                    "prompt": prompt,
                })
            results = query_openai_chat_model(
                engine=args.openai_engine,
                instances=instances,
                batch_size=args.eval_batch_size if args.eval_batch_size else 10,
                output_path=os.path.join(
                    args.save_dir, "predictions", f"{task_name}_openai_prediction_cache.jsonl"),
            )
            outputs = [result["output"] for result in results]

        targets = [example["target"] for example in task_examples]
        predictions = []
        for example, output in zip(task_examples, outputs):
            example["raw_output"] = output

            extracted_answer = re.search(r"So the answer is (.*?)\.", output)
            if extracted_answer:
                prediction = extracted_answer.group(1).strip()
            else:
                # only keep the first part of the output
                output = output.strip().split("\n\n")[0].strip()
                prediction = output.strip()

            example["prediction"] = prediction
            predictions.append(prediction)

        with open(os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl"), "w") as fout:
            for example in task_examples:
                fout.write(json.dumps(example) + "\n")

        assert len(predictions) == len(
            targets), "number of predictions and targets are not the same."
        performance[task_name] = exact_match.compute(
            predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]

        print(f"Task {task_name} - EM: {performance[task_name]}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        if len(performance) > 0:
            performance["average_exact_match"] = sum(performance.values()) / len(performance)
        else:
            performance["average_exact_match"] = 0.0
            print("[Warning] No tasks were evaluated, average score is 0.")
            
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bbh")
    parser.add_argument("--save_dir", type=str, default="results/bbh")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--openai_engine", type=str, default=None)
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument("--max_num_examples_per_task", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format")
    parser.add_argument("--convert_to_half", action="store_true")
    parser.add_argument("--convert_to_bf16", action="store_true")
    parser.add_argument("--subtask", type=int, default=None)
    parser.add_argument("--eval_valid", action="store_true")
    args = parser.parse_args()

    assert (args.model_name_or_path is None) != (
        args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
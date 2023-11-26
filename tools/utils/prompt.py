import os
import json
try:
    from human_eval.data import write_jsonl
except: # noqa
    write_jsonl = None


def filter_reply(text):
    text = text.split("\n\n")[0]
    if "```" in text:
        text = text.split("```")[1]
    if text.strip().startswith("def"):
        text = "\n".join(text.split("\n")[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text


def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


def generate_prompt(problem, task_type, dev_problems):
    if task_type == "human_eval":
        input_text = problem["prompt"]
    elif task_type == "base":
        input_text = problem["question"]
    elif task_type in ["cmmlu", "ceval"]:
        task_id = problem["task_id"]
        subject_phase = task_id.split("/")[0]
        pre_prompt = dev_problems[subject_phase]["prompt"]
        _ch_name = dev_problems[subject_phase]["_ch_name"]

        question = problem["question"]
        A, B, C, D = problem["A"], problem["B"], problem["C"], problem["D"]
        # answer = problem["answer"]
        if task_type in ["cmmlu"]:
            input_text = pre_prompt + \
                f"以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是: "
        elif task_type in ["ceval"]:
            input_text = pre_prompt + \
                f"以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: "
    return input_text


def text_postprocess(output, task_type):
    if task_type == "human_eval":
        output = filter_reply(output)
    elif task_type in ["cmmlu", "ceval"]:
        output = first_capital_postprocess(output)
    return output


def save_results(infer_file, samples, task_type):
    results = {}
    if task_type == "human_eval":
        write_jsonl(infer_file, samples)
    elif task_type in ["base", "cmmlu", "ceval"]:
        for idx in range(len(samples)):
            sample = samples[idx]
            task_id = sample["task_id"]
            if task_id not in results:
                results[task_id] = {}
            results[task_id]["input"] = sample["input"]
            results[task_id]["raw_output"] = sample["raw_output"]
            results[task_id]["output"] = sample["output"]
            results[task_id]["answer"] = sample["answer"]
            results[task_id]["infos"] = sample["infos"]

        with open(infer_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def evaluate(infer_file, task_type):
    if task_type == "human_eval":
        cmd = f"evaluate_functional_correctness {infer_file}"
        os.system(cmd)
    elif task_type == "base":
        all_count, all_accept_length = 0, 1
        out_tokens = [0, 0, 0, 0]
        with open(infer_file, "r") as f:
            results = json.load(f)
        for item in results:
            infos = results[item]["infos"]
            all_count += infos["count"]
            all_accept_length += infos["accept_length"]
            out_tokens[0] += infos.get("out_token_one", 0)
            out_tokens[1] += infos.get("out_token_two", 0)
            out_tokens[2] += infos.get("out_token_three", 0)
            out_tokens[3] += infos.get("out_token_four", 0)
        print(f"Forward: {all_count}; Accept_length: {all_accept_length}; Avg Accept_length: {all_accept_length / all_count}") # noqa
        print(f"Out Tokens([1, 2, 3, 4]): {out_tokens}")
    elif task_type in ["cmmlu", "ceval"]:
        correct, all_count, all_accept_length = 0, 0, 0
        out_tokens = [0, 0, 0, 0]
        with open(infer_file, "r") as f:
            results = json.load(f)
        for item in results:
            output = results[item]["output"]
            answer = results[item]["answer"]
            if output == answer:
                correct += 1
            infos = results[item]["infos"]
            all_count += infos["count"]
            all_accept_length += infos["accept_length"]
            out_tokens[0] += infos.get("out_token_one", 0)
            out_tokens[1] += infos.get("out_token_two", 0)
            out_tokens[2] += infos.get("out_token_three", 0)
            out_tokens[3] += infos.get("out_token_four", 0)
        print(f"Forward: {all_count}; Accept_length: {all_accept_length}; Avg Accept_length: {all_accept_length / all_count}")  # noqa
        print(f"Out Tokens([1, 2, 3, 4]): {out_tokens}")
        print(f"Correct: {correct}; Ratio: {correct / len(results)}")

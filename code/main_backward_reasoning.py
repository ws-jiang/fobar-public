import path_init

from tqdm import tqdm
import argparse
import json
import os
import copy

from utils.log_utils import LogUtils
from utils.parallel_utils import batch_get_api_merge
from utils.path_utils import PathUtils
import numpy as np

from utils.answer_clean_utils import answer_cleansing, string_number_dict

ds_path_dict = {
    "AddSub": "AddSub-cleaned",
    "SingleEq": "SingleEq-cleaned",
    "SVAMP": "SVAMP-cleaned",
    "MultiArith": "MultiArith-cleaned",
    "AQuA": "AQuA-cleaned",
    "GSM8K": "gsm8k_test-cleaned",
    "date": "date-cleaned",
    "letter": "last_letters-cleaned"
}

class BackwardReasoning():
    def __init__(self, args):
        self.args = args
        self.ds_name = args.ds
        self.temperature = args.temp
        self.threshold = args.threshold
        self.eng = args.eng
        self.num_repeat = args.num_repeat

        self.logger = LogUtils.get_or_init_logger(f"backward_cot_{self.args.method_name}_{self.ds_name}_{self.get_eng()}", "inv")

        self.inv_q_dict = {}
        self.inv_question_path = os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict[self.ds_name]}-backward-questions.json""")
        with open(self.inv_question_path) as f:
            inv_qs = json.load(f)
            self.logger.info(f"number of inverse question: {len(inv_qs)}")
            for e in inv_qs:
                if "inverse_question" in e:
                    if e["question"] not in self.inv_q_dict:
                        self.inv_q_dict[e["question"]] = []
                    self.inv_q_dict[e["question"]].append((e["inverse_question"], e['inverse_question_answer']))

        self.save_file = os.path.join(PathUtils.DATA_HOME_PATH,
                                      f"{ds_path_dict[self.ds_name]}_{self.args.method_name}_{self.get_eng()}_{self.args.num_samples}-backward-answers.json")
        if not self.args.cont:
            new_examples = []

            self.todo_path = os.path.join(PathUtils.DATA_HOME_PATH,
                                          f"{ds_path_dict[self.ds_name]}_{self.args.method_name}_answer_{self.get_eng()}_{self.args.num_samples}_stat.json")
            with open(self.todo_path) as f:
                self.examples = json.load(f)
                for e in self.examples:
                    question = e['question']
                    ans_stat = e['ans_stat']
                    candidate_answers = [k for k in ans_stat if ans_stat[k] >= self.threshold]

                    if len(candidate_answers) > 1 and question in self.inv_q_dict:
                        for candidate_answer in candidate_answers:
                            for temp_inv_e in self.inv_q_dict[e["question"]]:
                                new_e = copy.deepcopy(e)
                                new_e["candidate_answer"] = candidate_answer
                                new_e["inv_question"] = temp_inv_e[0]
                                if temp_inv_e[1] in string_number_dict:
                                    new_e["inv_question_ans"] = str(string_number_dict[temp_inv_e[1]])
                                else:
                                    new_e['inv_question_ans'] = temp_inv_e[1]
                                new_examples.append(new_e)
                    else:
                        new_examples.append(e)
                self.examples = np.repeat(new_examples, self.args.num_repeat).tolist()
                self.save_data()

        with open(self.save_file) as f:
            self.examples = json.load(f)

        self.unknown_var = "x"
        if self.ds_name in ("AQuA"):
            self.prompt = self.get_prompt("backward_cot_aqua.txt")
        elif self.ds_name in ("GSM8K", "AddSub", "SVAMP", "SingleEq", "MultiArith"):
            self.prompt = self.get_prompt("backward_cot_gsm8k.txt")
        elif self.ds_name in ("date", "letter"):
            self.prompt = self.get_prompt(f"backward_cot_{self.ds_name}.txt")
        else:
            raise ValueError(f"unknown dataset: {self.ds_name}")

    def get_eng(self):
        if "gpt-4" in self.eng:
            return "gpt-4"
        elif "gpt-3.5-turbo" in self.eng:
            return "gpt-3.5-turbo"
        else:
            return self.eng

    def save_data(self):
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=4)

    def get_prompt(self, prompt_file_name):
        prompt_file = os.path.join(PathUtils.CONFIG_HOME_PATH, prompt_file_name)
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def evaluate(self, end_idx):
        result_stat_dict = {}
        for e in self.examples[0:end_idx]:
            question = e['question']

            if question not in result_stat_dict:
                result_stat_dict[question] = []

            result_stat_dict[question].append(e)

        num_correct = 0
        sc_method_cnt = 0
        inv_method_cnt = 0
        for q in result_stat_dict:
            e = result_stat_dict[q][0]
            answer = e['answer']
            if "inv_question" not in result_stat_dict[q][0]:
                pred_ans = max(e["ans_stat"], key=e["ans_stat"].get)
                sc_method_cnt += 1
            else:
                e_list = result_stat_dict[q]
                inv_ans = [answer_cleansing(_['inv_question_ans'], self.ds_name) for _ in e_list]
                for idx, _ in enumerate(inv_ans):
                    if _ in string_number_dict:
                        inv_ans[idx] = string_number_dict[_]
                pred_inv_answers = [_['inv_question_pred_answer'] for _ in e_list]
                inv_correct_counter = {}
                for idx, temp_e in enumerate(e_list):
                    if temp_e["candidate_answer"] not in inv_correct_counter:
                        inv_correct_counter[temp_e["candidate_answer"]] = 0
                    if inv_ans[idx] == pred_inv_answers[idx]:
                        inv_correct_counter[temp_e["candidate_answer"]] += 1
                values = np.array(inv_correct_counter.values())

                if np.sum(values == values.max(keepdims=True)) == 1:
                    inv_method_cnt += 1
                    pred_ans = max(inv_correct_counter, key=inv_correct_counter.get)
                else:
                    pred_ans = max(e["ans_stat"], key=e["ans_stat"].get)
                    sc_method_cnt += 1

            if pred_ans == answer:
                num_correct += 1
        self.logger.info(f"#sc method: {sc_method_cnt}; #inv method: {inv_method_cnt}")
        return num_correct, len(result_stat_dict.keys()), num_correct / len(result_stat_dict.keys())

    def get_inv_split_str(self):
        return f"The value of {self.unknown_var} is"

    def fetch_data_from_openai(self):
        if self.ds_name == "AQuA":
            def wrap(e):
                option_idx_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
                return f"""{self.prompt}\n\nQuestion: {e['inv_question']}\nIf we know the answer of the above question is {e['options'][option_idx_dict[e['answer']]]}, what is the value of unknown variable {self.unknown_var}?\nA:"""
        else:
            def wrap(e):
                return f"""{self.prompt}\n\nQuestion: {e['inv_question']}\nIf we know the answer of the above question is {e['candidate_answer']}, what is the value of unknown variable {self.unknown_var}?\nA:"""

        def extract(e, reply):
            e['inv_question_pred_answer'] = reply
            e['pred_inv_answer_cleaned'] = answer_cleansing(pred=reply, ds_name=self.ds_name, split_str=self.get_inv_split_str())

        todo_list = []
        for i, example in tqdm(enumerate(self.examples), total=len(self.examples)):
            if i % 10 == 0:
                self.logger.info(f"processing: {i}/{len(self.examples)}")

            if "inv_question_pred_answer" in example or 'inv_question' not in example:
                self.logger.info(f"skip {i}th question, has no inv question.")
                continue

            todo_list.append(example)

            if (len(todo_list) >= self.args.batch_size) or i >= (len(self.examples) - 1):
                if len(todo_list) > 0:
                    batch_get_api_merge(examples=todo_list, eng=self.args.eng, pre_fun=wrap, post_fun=extract,
                                        logger=self.logger, n_processes=self.args.num_proc,
                                        temperature=self.temperature, timeout=self.args.time_out, max_try=0)
                    todo_list = []
                    self.save_data()
                    num_correct, num_examples, acc = self.evaluate(i + 1)
                    self.logger.info(
                        "=" * 20 + f"processed: {i}/{len(self.examples)}, acc: {num_correct}/{num_examples}={100 * acc:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--eng', default="text-davinci-003", type=str)
    parser.add_argument('--ds', default="AddSub", type=str)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--method_name', default="SCCoT", type=str)
    parser.add_argument('--num_repeat', default=20, type=int)
    parser.add_argument('--threshold', default=1, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--time_out', default=30, type=int)
    parser.add_argument('--num_samples', default=0, type=int)
    parser.add_argument('--num_proc', default=16, type=int)
    args = parser.parse_args()

    # for these two models, max batch size is 20
    if args.eng in ("text-davinci-003"):
        args.batch_size = 20

    rephrase_cot = BackwardReasoning(args)
    rephrase_cot.fetch_data_from_openai()
import path_init

import os
import json
import numpy as np
import argparse

from utils.path_utils import PathUtils
from utils.answer_clean_utils import answer_cleansing

parser = argparse.ArgumentParser()

parser.add_argument('--eng', default="text-davinci-003", type=str)
parser.add_argument('--method_name', default="SCComplexCoT", type=str)
parser.add_argument('--ds_name', default="MultiArith", type=str)
parser.add_argument('--MB', default=8, type=int)
args = parser.parse_args()

model_name = args.eng
method = args.method_name
MB = args.MB
ds_name = args.ds_name

print(model_name, method)

file_path_dict = {
    "SingleEq": f"SingleEq-cleaned_{method}_{model_name}_0-backward-answers.json",
    "MultiArith": f"MultiArith-cleaned_{method}_{model_name}_0-backward-answers.json",
    "AddSub": f"AddSub-cleaned_{method}_{model_name}_0-backward-answers.json",
    "GSM8K": f"gsm8k_test-cleaned_{method}_{model_name}_0-backward-answers.json",
    "SVAMP": f"SVAMP-cleaned_{method}_{model_name}_0-backward-answers.json",
    "AQuA": f"AQuA-cleaned_{method}_{model_name}_0-backward-answers.json",
}

ds_path = os.path.join(PathUtils.DATA_HOME_PATH, file_path_dict[ds_name])

with open(ds_path) as f:
    examples = json.load(f)

question_list = []
for _e in examples:
    q = _e["question"]
    if q not in question_list:
        question_list.append(q)

result_stat_dict = {}
for e in examples:
    question = e['question']

    if question not in result_stat_dict:
        result_stat_dict[question] = []

    result_stat_dict[question].append(e)

sc_dict = {q: dict(ans_stat=result_stat_dict[q][0]['ans_stat'], gt=result_stat_dict[q][0]['answer']) for q in
           question_list}

merged_pred_dict = {}
for q in question_list:
    forward_prob = sc_dict[q]['ans_stat']
    total_forward_prob = (sum(forward_prob.values()))
    total_forward_prob = {k: v / total_forward_prob for k, v in forward_prob.items()}
    sc_ans = max(total_forward_prob, key=total_forward_prob.get)
    gt = sc_dict[q]['gt']

    merged_pred_dict[q] = dict(gt=gt, sc_ans=sc_ans, ans_stat=total_forward_prob)

backward_answer_prob = {}
eps = 1e-8
for q in question_list:
    temp_e = result_stat_dict[q][0]
    has_inv_question = np.array([1 for _ in result_stat_dict[q] if "inv_question" in _]).sum() > 0
    sc_pred_ans = max(temp_e["ans_stat"], key=temp_e["ans_stat"].get)
    gt = temp_e['answer']

    temp_dict = {}
    if has_inv_question:
        inq_dict = {}
        for e in result_stat_dict[q]:
            key_inq_candidate_ans = f"""{e['inv_question']}##{e['candidate_answer']}"""
            if key_inq_candidate_ans not in inq_dict:
                inq_dict[key_inq_candidate_ans] = []
            inq_dict[key_inq_candidate_ans].append(e)

        for inv_q_ca in inq_dict:
            for e in inq_dict[inv_q_ca][0:MB]:
                candidate_answer = e['candidate_answer']
                if candidate_answer not in temp_dict:
                    temp_dict[candidate_answer] = 0

                if "inv_question" not in e:
                    continue

                split_str = "The value of K is" if ds_name == "AQuA" else "The value of x is"
                pred_inv_ans = answer_cleansing(e['inv_question_pred_answer'], ds_name=ds_name, split_str=split_str)
                inv_question_ans = e['inv_question_ans']
                if pred_inv_ans == inv_question_ans:
                    temp_dict[candidate_answer] += 1
        inv_pred_ans = max(temp_dict, key=temp_dict.get)
        total_backward_prob = (sum(temp_dict.values()))
        total_backward_prob = {k: (v + eps / len(temp_dict)) / (total_backward_prob + eps) for k, v in
                               temp_dict.items()}
        backward_answer_prob[q] = dict(ans_stat=total_backward_prob, inv_pred_ans=inv_pred_ans, gt=gt)

final_dict = {}

merged_accs, forward_accs, backward_accs = [], [], []
alpha = 0.5
beta = 1 - alpha
for q in question_list:
    gt = merged_pred_dict[q]['gt']
    sc_ans_stat = merged_pred_dict[q]['ans_stat']
    sc_ans = merged_pred_dict[q]['sc_ans']

    merged_ans_stat = {}
    if q in backward_answer_prob:
        inv_pred_ans = backward_answer_prob[q]['inv_pred_ans']
        inv_pred_prob = backward_answer_prob[q]['ans_stat']
        for ans in sc_ans_stat:
            merged_ans_stat[ans] = (sc_ans_stat[ans]) ** alpha * (
                    backward_answer_prob[q]['ans_stat'].get(ans, eps)) ** beta
        final_dict[q] = dict(gt=gt, sc_pred_prob=sc_ans_stat, sc_pred_ans=sc_ans,
                             merged_pred_prob=merged_ans_stat, inv_pred_ans=inv_pred_ans, inv_pred_prob=inv_pred_prob)
    else:
        merged_ans_stat = sc_ans_stat

        final_dict[q] = dict(gt=gt, sc_pred_prob=sc_ans_stat, sc_pred_ans=sc_ans, inv_pred_ans=sc_ans,
                             merged_pred_prob=merged_ans_stat)

    merged_pred_ans = max(merged_ans_stat, key=merged_ans_stat.get)
    final_dict[q]['merged_pred_ans'] = merged_pred_ans

acc_1 = len([_ for _ in final_dict if final_dict[_]['gt'] == final_dict[_]['sc_pred_ans']]) / len(final_dict)
acc_2 = len([_ for _ in final_dict if final_dict[_]['gt'] == final_dict[_]['inv_pred_ans']]) / len(final_dict)
acc_3 = len([_ for _ in final_dict if final_dict[_]['gt'] == final_dict[_]['merged_pred_ans']]) / len(final_dict)
print(f"{ds_name} (alpha={alpha:.2f}): sc={100 * acc_1:.2f}; inv={100 * acc_2:.2f}; merged={100 * acc_3:.2f}")

merged_accs.append(acc_3)
backward_accs.append(acc_2)
forward_accs.append(acc_1)

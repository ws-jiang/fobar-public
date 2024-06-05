import re


string_number_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                      "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def str_minus_1(s):
    def cycle_the_order(order):
        if order in [65, 97]:
            return order + 25
        else:
            return order - 1
    return "".join([chr(cycle_the_order(ord(_))) for _ in s])

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n

def ans_clean_date_x(pred, split_str):
    preds = pred.split(split_str)

    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]

        # If there is no candidate in list, null is set.
        if len(pred) == 0:
            pred = ""
        else:
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred[-1] == "/":
                pred = pred[:-1]
        return pred
    else:
        preds = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+', pred)]
        if len(preds) > 0:
            return preds[-1]
        else:
            return ""

def answer_cleansing(pred, ds_name, split_str="The answer is"):
    preds = pred.split(split_str)

    answer_flag = True if len(preds) > 1 else False

    if ds_name in ("AQuA"):
        if len(preds) > 0:
            pred = re.findall(r'A|B|C|D|E', preds[-1].upper())
            if len(pred) > 0:
                if answer_flag:
                    # choose the first element in list ...
                    pred = pred[0]
                else:
                    # choose the last element in list ...
                    pred = pred[-1]
                return pred
            else:
                return ""
        return ""

    pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    return pred
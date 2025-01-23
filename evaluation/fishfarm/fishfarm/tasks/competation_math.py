from dataclasses import dataclass
from math import isclose
from typing import Any, Iterable, Optional, Sequence, Union

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a: str = string.split("/")[0]
    b: str = string.split("/")[1]
    try:
        a_int: int = int(a)
        b_int: int = int(b)
        assert string == "{}/{}".format(a_int, b_int)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (AssertionError, ValueError):
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")

    string = string.replace("\\!", "")

    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")

    string = _remove_right_units(string)

    string = string.replace(r"\\%", "")
    string = string.replace(r"\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)

    string = string.replace(" ", "")

    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)

    return string


def is_digit(s: Union[bool, float, str]) -> bool:
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def symbolic_equal(a: str, b: str) -> bool:
    def _parse(s: str) -> Any:
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except Exception:
        pass
    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    if (
        (prediction.startswith("[") and prediction.endswith("]"))
        and (reference.startswith("[") and reference.endswith("]"))
        or (prediction.startswith("(") and prediction.endswith(")"))
        and (reference.startswith("(") and reference.endswith(")"))
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if symbolic_equal(prediction, reference):
        return True

    return False


def is_equiv(str1: Optional[str], str2: Optional[str]) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return math_equal(ss1, ss2) or ss1 == ss2
    except (AssertionError, TypeError, ValueError):
        return math_equal(str1, str2) or str1 == str2


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx: Optional[int] = None

    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        assert right_brace_idx is not None
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: Optional[str]) -> Optional[str]:
    left = "\\boxed{"
    if s is None:
        return None
    else:
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        except (AssertionError, TypeError, ValueError):
            return None


@dataclass
class MathSample:

    problem: str
    answer: Optional[str] = None
    type: Optional[str] = None


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count


def extract_ans(completion: str) -> Optional[str]:

    split_ans = completion.split("The answer is: ")
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split(".\n")[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        return extract_ans
    else:
        return remove_boxed(last_boxed_only_string(completion))


class LatexFormatMathTask(Task):
    def __init__(
        self,
        samples: Sequence[MathSample],
        context_messages: Sequence[Message] = (),
    ):
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.problem))
            requests.append(GenerationRequest(messages=messages, task_name="MATH"))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_ans(output)

            sample_details.append(
                dict(
                    problem=sample.problem,
                    output=output,
                    answer=sample.answer,
                    type=sample.type,
                    prediction=prediction,
                    correct=is_equiv(sample.answer, prediction),
                )
            )

        aggregate_metrics = {
            "acc": mean(
                float(sd["correct"]) if isinstance(sd["correct"], (bool)) else 0.0
                for sd in sample_details
            )
        }

        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )

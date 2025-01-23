import random
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


def extract_answer(text: str) -> Optional[str]:
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text: str) -> Optional[str]:
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text: str) -> Optional[str]:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def is_correct(pred: Optional[str], answer: str, options: list[str]) -> bool:
    if not pred:
        random.seed(42)
        x = random.randint(0, len(options) - 1)
        if ["A", "B", "C", "D", "E"][x] == answer:
            return True
        else:
            return False
    elif pred == answer:
        return True
    else:
        return False


@dataclass
class Ai2ArcSample:

    question: str
    question_id: str
    options: list[str]
    answer: str


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count


class Ai2ArcTask(Task):
    def __init__(
        self,
        samples: Sequence[Ai2ArcSample],
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
            messages.append(Message(role="user", content=sample.question))
            requests.append(GenerationRequest(messages=messages, task_name="ARC"))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_answer(result.generation)

            sample_details.append(
                dict(
                    problem=sample.question,
                    output=output,
                    answer=sample.answer,
                    prediction=prediction,
                    correct=is_correct(prediction, sample.answer, sample.options),
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

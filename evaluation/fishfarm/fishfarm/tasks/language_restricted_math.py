import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import huggingface_hub

from ..imports import try_import
from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult

with try_import() as _imports:
    import fasttext

_imports.check()


@dataclass
class MathSample:

    problem: str
    answer: int


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count


def extract_answer_number(completion: str) -> Optional[float]:
    matches = re.findall(r"\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))


class LanguageRestrictedMathTask(Task):
    def __init__(
        self,
        samples: Sequence[MathSample],
        context_messages: Sequence[Message] = (),
        languages: Sequence[str] = ("ja", "en"),
    ):
        self.samples = list(samples)
        self.languages = languages
        self.context_messages = context_messages
        if len(self.languages) != 0:
            lid176ftz_path = huggingface_hub.hf_hub_download(
                "julien-c/fasttext-language-id", "lid.176.ftz"
            )
            self.lid_model = fasttext.load_model(lid176ftz_path)

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
            requests.append(GenerationRequest(messages=messages, task_name="GSM8K"))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_answer_number(result.generation)
            if len(self.languages) != 0:
                lid_probs = dict(
                    zip(*self.lid_model.predict(output.replace("\n", ""), k=-1))
                )

            sample_details.append(
                dict(
                    problem=sample.problem,
                    output=output,
                    answer=sample.answer,
                    prediction=prediction,
                    correct=sample.answer == prediction,
                    **{
                        f"lang_{lang}": lid_probs.get(f"__label__{lang}", 0.0)
                        for lang in self.languages
                    },
                )
            )

        aggregate_metrics = {"acc": mean(sd["correct"] for sd in sample_details)}
        for lang in self.languages:
            aggregate_metrics[f"acc_{lang}"] = mean(
                (sd["correct"] and sd[f"lang_{lang}"] > 0.5) for sd in sample_details
            )

        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )

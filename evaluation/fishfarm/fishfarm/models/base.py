from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

Role = Literal["system", "user", "assistant", "assistant_prefill"]


@dataclass
class Message:

    role: Role
    content: str


@dataclass
class GenerationRequest:

    messages: list[Message]

    max_tokens: Optional[int] = None
    stop: Sequence[str] = ()
    
    task_name: Optional[str] = None


@dataclass
class GenerationResult:

    request: GenerationRequest
    generation: str

    task_name: Optional[str] = None

@dataclass
class NLLRequest:

    messages: list[Message]


@dataclass
class NLLResult:

    request: NLLRequest
    sum_nll: float
    num_considered_tokens: int


class Model:

    def generate(
        self, requests: Sequence[GenerationRequest]
    ) -> Iterable[GenerationResult]:
        raise NotImplementedError()

    def nll(self, requests: Sequence[NLLRequest]) -> Iterable[NLLResult]:
        raise NotImplementedError()

import copy
import dataclasses
from typing import Any, Iterable, Optional, Sequence

from fishfarm.models.base import NLLRequest, NLLResult
from transformers import PreTrainedTokenizerBase

from ..imports import try_import
from .base import GenerationRequest, GenerationResult, Message, Model
from .tokenization_utils import tokenize_messages

with try_import() as _imports:
    import vllm

_imports.check()


class VLLMModel(Model):

    def __init__(
        self,
        llm: vllm.LLM,
        sampling_params: vllm.SamplingParams,
        chat_template: Optional[str],
    ) -> None:
        self.llm = llm
        self.chat_template = chat_template
        self.sampling_params = sampling_params

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = self.llm.get_tokenizer()

        if not hasattr(tokenizer, "apply_chat_template"):
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer
            else:
                raise ValueError(
                    "The tokenizer does not have the 'apply_chat_template' method. "
                    "This is likely because of the versions of vLLM or transformers."
                )

        return tokenizer

    def _into_prompt(self, messages: Sequence[Message]) -> str:
        tokenizer = self.get_tokenizer()
        prefill_text = ""
        n_assistant_prefill = sum([m.role == "assistant_prefill" for m in messages])
        if n_assistant_prefill > 1:
            raise ValueError(
                f"There must be at most one assistant_prefill role, but got {n_assistant_prefill}",
            )
        if n_assistant_prefill:
            assert (
                messages[-1].role == "assistant_prefill"
            ), "assistant_prefill role must be the last message"
            prefill_text = messages[-1].content
            messages = messages[:-1]
        prompt: str = tokenizer.apply_chat_template(
            conversation=[dataclasses.asdict(message) for message in messages],
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += prefill_text
        return prompt

    def _predict_log_probs(self, token_ids_list: list[list[int]]) -> list[list[float]]:
        sampling_params = copy.copy(self.sampling_params)
        sampling_params.prompt_logprobs = 1
        sampling_params.max_tokens = 1

        completions = self.llm.generate(
            prompt_token_ids=token_ids_list,
            sampling_params=sampling_params,
        )

        log_probs_list = []
        for token_ids, completion in zip(token_ids_list, completions):
            log_probs = []
            assert completion.prompt_logprobs is not None
            assert token_ids == completion.prompt_token_ids
            assert len(token_ids) == len(completion.prompt_logprobs)
            for token_id, logprob_dict in zip(token_ids, completion.prompt_logprobs):
                if logprob_dict is None:
                    log_probs.append(0.0)
                else:
                    logprob_entry: Any = logprob_dict[token_id]

                    if isinstance(logprob_entry, float):
                        log_probs.append(logprob_entry)
                    else:
                        log_probs.append(logprob_entry.logprob)

            log_probs_list.append(log_probs)

        return log_probs_list

    def generate(
        self, requests: Sequence[GenerationRequest]
    ) -> Iterable[GenerationResult]:

        prompts = [self._into_prompt(request.messages) for request in requests]
        completions = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
        )

        for request, completion in zip(requests, completions):
            yield GenerationResult(
                request=request, generation=completion.outputs[0].text
            )

    def nll(self, requests: Sequence[NLLRequest]) -> Iterable[NLLResult]:
        masked_tokens_list = [
            tokenize_messages(
                request.messages, self.get_tokenizer(), self.chat_template
            )
            for request in requests
        ]
        log_probs_list = self._predict_log_probs(
            [masked_tokens.token_ids for masked_tokens in masked_tokens_list]
        )

        results = []
        for log_probs, masked_tokens, request in zip(
            log_probs_list, masked_tokens_list, requests
        ):
            assert len(log_probs) == len(masked_tokens.mask)

            sum_nll = 0.0
            num_considered_tokens = 0
            for log_prob, mask_value in zip(log_probs, masked_tokens.mask):
                if mask_value:
                    sum_nll += -log_prob
                    num_considered_tokens += 1

            results.append(
                NLLResult(
                    request=request,
                    sum_nll=sum_nll,
                    num_considered_tokens=num_considered_tokens,
                )
            )

        return results




class MoE_VLLMModel(Model):

    def __init__(
        self,
        llm: vllm.LLM,
        sampling_params: vllm.SamplingParams,
        chat_template_dict: dict,
    ) -> None:
        self.llm = llm
        self.chat_template_dict = chat_template_dict
        self.sampling_params = sampling_params

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = self.llm.get_tokenizer()

        if not hasattr(tokenizer, "apply_chat_template"):
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer
            else:
                raise ValueError(
                    "The tokenizer does not have the 'apply_chat_template' method. "
                    "This is likely because of the versions of vLLM or transformers."
                )

        return tokenizer

    def _into_prompt(self, messages: Sequence[Message], task_name: str) -> str:
        tokenizer = self.get_tokenizer()
        prefill_text = ""
        n_assistant_prefill = sum([m.role == "assistant_prefill" for m in messages])
        if n_assistant_prefill > 1:
            raise ValueError(
                f"There must be at most one assistant_prefill role, but got {n_assistant_prefill}",
            )
        if n_assistant_prefill:
            assert (
                messages[-1].role == "assistant_prefill"
            ), "assistant_prefill role must be the last message"
            prefill_text = messages[-1].content
            messages = messages[:-1]
        
        task_name_to_chat_template = {
            "mbpp": self.chat_template_dict["mbpp2"],
            "ARC": self.chat_template_dict["arc"],
            "GSM8K": self.chat_template_dict["gsm8k"],
        }
        try:
            chat_template = task_name_to_chat_template[task_name]
        except:
            raise ValueError(f"Unknown task_name: {task_name} in MoE_VLLMModel")
        prompt: str = tokenizer.apply_chat_template(
            conversation=[dataclasses.asdict(message) for message in messages],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += prefill_text
        return prompt

    def _predict_log_probs(self, token_ids_list: list[list[int]]) -> list[list[float]]:
        import ipdb; ipdb.set_trace()
        sampling_params = copy.copy(self.sampling_params)
        sampling_params.prompt_logprobs = 1
        sampling_params.max_tokens = 1

        completions = self.llm.generate(
            prompt_token_ids=token_ids_list,
            sampling_params=sampling_params,
        )

        log_probs_list = []
        for token_ids, completion in zip(token_ids_list, completions):
            log_probs = []
            assert completion.prompt_logprobs is not None
            assert token_ids == completion.prompt_token_ids
            assert len(token_ids) == len(completion.prompt_logprobs)
            for token_id, logprob_dict in zip(token_ids, completion.prompt_logprobs):
                if logprob_dict is None:
                    log_probs.append(0.0)
                else:
                    logprob_entry: Any = logprob_dict[token_id]

                    if isinstance(logprob_entry, float):
                        log_probs.append(logprob_entry)
                    else:
                        log_probs.append(logprob_entry.logprob)

            log_probs_list.append(log_probs)

        return log_probs_list

    def generate(
        self, requests: Sequence[GenerationRequest]
    ) -> Iterable[GenerationResult]:

        prompts = [self._into_prompt(request.messages, request.task_name) for request in requests]
        completions = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
        )

        for request, completion in zip(requests, completions):
            yield GenerationResult(
                request=request, generation=completion.outputs[0].text, task_name=request.task_name
            )

    def nll(self, requests: Sequence[NLLRequest]) -> Iterable[NLLResult]:
        import ipdb; ipdb.set_trace()
        masked_tokens_list = [
            tokenize_messages(
                request.messages, self.get_tokenizer(), self.chat_template
            )
            for request in requests
        ]
        log_probs_list = self._predict_log_probs(
            [masked_tokens.token_ids for masked_tokens in masked_tokens_list]
        )

        results = []
        for log_probs, masked_tokens, request in zip(
            log_probs_list, masked_tokens_list, requests
        ):
            assert len(log_probs) == len(masked_tokens.mask)

            sum_nll = 0.0
            num_considered_tokens = 0
            for log_prob, mask_value in zip(log_probs, masked_tokens.mask):
                if mask_value:
                    sum_nll += -log_prob
                    num_considered_tokens += 1

            results.append(
                NLLResult(
                    request=request,
                    sum_nll=sum_nll,
                    num_considered_tokens=num_considered_tokens,
                )
            )

        return results

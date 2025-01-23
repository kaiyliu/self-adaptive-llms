import os
import re
from dataclasses import dataclass
from typing import Iterable, Tuple

import datasets
import fishfarm
import vllm
from omegaconf import DictConfig
from fishfarm.models.vllm_model import VLLMModel, MoE_VLLMModel
from fishfarm.tasks.base import TaskResult
from fishfarm.tasks.evalplus import load_dataset
from typing import Union
from .base import Task, get_download_dir
import hydra


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count


def extract_ans(text):
    """Fetch the string within \\boxed{}."""
    match = re.search(r"\\boxed{([^}]*)}", text)
    if match:
        return match.group(1)  # Return the content inside the \boxed{}
    else:
        return None  # Return None if no match is found


@dataclass
class CategorySample:
    question: str
    label: str


class CategoryClassficiationTask(fishfarm.tasks.base.Task):
    def __init__(
        self,
        samples,
        context_messages,
    ):
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model,
        sample_ids,
    ):
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(fishfarm.Message(role="user", content=sample.question))
            requests.append(fishfarm.models.GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_ans(output)

            sample_details.append(
                dict(
                    question=sample.question,
                    label=sample.label,
                    output=output,
                    prediction=prediction,
                    correct=sample.label == prediction,
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


class Gsm8k_Arc_Mbpp2_Task(Task):
    def __init__(
        self,
        wrapped_gsm8k_task: Union[Task, DictConfig],
        wrapped_arc_task: Union[Task, DictConfig],
        wrapped_mbpp2_task: Union[Task, DictConfig],
    ):
        self.debug_num = 10
        print(f"\n[debug_num]: {self.debug_num}\n")
        # 初始化
        if isinstance(wrapped_gsm8k_task, Task):
            self.wrapped_gsm8k_task: Task = wrapped_gsm8k_task
        else:
            self.wrapped_gsm8k_task: Task = hydra.utils.instantiate(wrapped_gsm8k_task)
        if isinstance(wrapped_arc_task, Task):
            self.wrapped_arc_task: Task = wrapped_arc_task
        else:
            self.wrapped_arc_task: Task = hydra.utils.instantiate(wrapped_arc_task)
        if isinstance(wrapped_mbpp2_task, Task):
            self.wrapped_mbpp2_task: Task = wrapped_mbpp2_task
        else:
            self.wrapped_mbpp2_task: Task = hydra.utils.instantiate(wrapped_mbpp2_task)

        self.wrapped_gsm8k_task.debug_num = self.debug_num
        self.wrapped_arc_task.debug_num = self.debug_num
        self.wrapped_mbpp2_task.debug_num = self.debug_num

        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": {
                "gsm8k": self.wrapped_gsm8k_task.model_to_template["meta-llama/Meta-Llama-3-8B-Instruct"],
                "arc": self.wrapped_arc_task.model_to_template["meta-llama/Meta-Llama-3-8B-Instruct"],
                "mbpp2": self.wrapped_mbpp2_task.model_to_template["meta-llama/Meta-Llama-3-8B-Instruct"],
            },
            "mistralai/Mistral-7B-Instruct-v0.3": {
                "gsm8k": self.wrapped_gsm8k_task.model_to_template["mistralai/Mistral-7B-Instruct-v0.3"],
                "arc": self.wrapped_arc_task.model_to_template["mistralai/Mistral-7B-Instruct-v0.3"],
                "mbpp2": self.wrapped_mbpp2_task.model_to_template["mistralai/Mistral-7B-Instruct-v0.3"],
            },
        }
        self.system_msg = {
            "gsm8k": self.wrapped_gsm8k_task.system_msg,
            "arc": self.wrapped_arc_task.system_msg,
            "mbpp2": self.wrapped_mbpp2_task.system_msg,
        }
        self.target_metric_train = {
            "gsm8k": self.wrapped_gsm8k_task.target_metric_train,
            "arc": self.wrapped_arc_task.target_metric_train,
            "mbpp2": self.wrapped_mbpp2_task.target_metric_train,
        }
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = {
            "gsm8k": self.wrapped_gsm8k_task.target_metric_transfer,
            "arc": self.wrapped_arc_task.target_metric_transfer,
            "mbpp2": self.wrapped_mbpp2_task.target_metric_transfer,
        }
        self.has_transfer_split = True
        self.has_training_split = True
        
        # get dataset
        gsm8k_train_evaluator, gsm8k_test_evaluator = self.wrapped_gsm8k_task.get_evaluator()
        arc_train_evaluator, arc_test_evaluator, arc_transfer_evaluator = self.wrapped_arc_task.get_evaluator()
        mbpp2_train_evaluator, mbpp2_test_evaluator, mbpp2_transfer_evaluator = self.wrapped_mbpp2_task.get_evaluator()
        
        self.train_samples = gsm8k_train_evaluator.samples + arc_train_evaluator.samples + mbpp2_train_evaluator.samples
        self.train_evaluators = {
            "gsm8k": gsm8k_train_evaluator,
            "arc": arc_train_evaluator,
            "mbpp2": mbpp2_train_evaluator,
        }
        self.test_evaluators = {
            "gsm8k": gsm8k_test_evaluator,
            "arc": arc_test_evaluator,
            "mbpp2": mbpp2_test_evaluator,
        }
        self.transfer_evaluators = {
            "arc": arc_transfer_evaluator,
            "mbpp2": mbpp2_transfer_evaluator,
        }

    def get_train_data(self=400):
        train_ix = range(0, len(self.train_samples), 2)
        valid_ix = range(1, len(self.train_samples), 2)
        return self.train_samples, train_ix, valid_ix

    def get_rewards(self, res):
        if "base_correct" in res.sample_details[0]:
            rewards = [1.0 if x["base_correct"] else -1.0 for x in res.sample_details]
        else:
            rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        return self.train_evaluators, self.test_evaluators, self.transfer_evaluators

    def get_prompt(self, tokenizer, samples, ix, model_id):
        # # 获取样本的类名
        # samples_class_name = []
        # for sample_i in samples:
        #     samples_class_name.append(sample_i.__class__.__name__)
        samples_class_name_to_task ={
            'TextToCodeProblem': self.wrapped_mbpp2_task,
            'Ai2ArcSample': self.wrapped_arc_task,
            'MathSample': self.wrapped_gsm8k_task
        }
        try:
            return samples_class_name_to_task[samples[ix].__class__.__name__].get_prompt(tokenizer, samples, ix, model_id)
        except:
            import ipdb; ipdb.set_trace()

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=2048,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")),
            # download_dir=get_download_dir(),
        )
        chat_template_dict = self.model_to_template[model_id]
        # This may change with vLLM versions.
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        vllm_model = MoE_VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=1024,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template_dict=chat_template_dict,
        )
        return vllm_model

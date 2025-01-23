import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from logging_utils import get_mean_std_max_min_dict
from utils import (backward, eval_model, forward, load_base_params,
                   load_hf_params_to_vllm)
from evaluation.fishfarm.fishfarm.tasks.base import TaskResult


class OptimizationAlgorithm(abc.ABC):
    def __init__(self, **kwargs):
        nn.Module.__init__(self=self)

    @abc.abstractmethod
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        original_model_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        raise NotADirectoryError

    @abc.abstractmethod
    def update(self, policy):
        raise NotImplementedError

    def log_optim(self, metrics_to_log):
        pass


class Reinforce(OptimizationAlgorithm, nn.Module):
    def __init__(
        self, policy, gpu, max_grad_norm, lr, rw_norm, rw_clip, kl_ref_coeff, **kwargs
    ):
        nn.Module.__init__(self=self)
        self.gpu = gpu
        self.kl_ref_coeff = kl_ref_coeff
        self.use_kl_loss = kl_ref_coeff > 0.0
        self.max_grad_norm = float(max_grad_norm)
        self.lr = lr
        self.rw_norm = rw_norm
        self.rw_clip = rw_clip
        self.optimizer = torch.optim.Adam(policy.trainable_params, lr=lr)

    def compute_ref_logprobs(
        self,
        model,
        tokenizer,
        prompts,
        res,
    ):
        ref_log_probs_list = []
        print("Computing reference log probs...")
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + res.sample_details[j]["output"],
                return_tensors="pt",
            ).input_ids.to(self.gpu)
            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            ref_log_probs_list.append(log_probs.detach().cpu())
        return ref_log_probs_list

    def get_rewards(self, task_loader, res):
        rw_norm = self.rw_norm
        rw_clip = self.rw_clip
        rewards = task_loader.get_rewards(res=res)

        if rw_norm:
            rewards = np.array(rewards)
            mean_rw = np.mean(rewards)
            std_rw = np.clip(np.std(rewards), a_min=1e-7, a_max=None)
            rewards = (rewards - mean_rw) / std_rw
        if rw_clip is not None:
            if rw_clip > 0:
                rewards = np.array(rewards)
                rewards = np.clip(rewards, a_min=-rw_clip, a_max=rw_clip)
        return rewards

    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        original_model_params,
        metrics_to_log,
        vllm_model=None,
        moe_mode=False,
        **kwargs,
    ):
        use_kl_loss = self.use_kl_loss
        kl_ref_coeff = self.kl_ref_coeff

        gpu = self.gpu

        # 1. 加载当前batch的问题
        prompts = [
            task_loader.get_prompt(
                tokenizer,
                train_data,
                i,
                model_id=model_id,
            )
            for i in batch_ix
        ]
        clipped_batch_size = len(prompts)

        # 2. 加载当前参数
        learnable_params = policy.get_learnable_params()
        new_params = forward(
            policy, model, base_params, decomposed_params, learnable_params
        )
        
        # 3. 加载当前参数到vllm，采样，并计算reward
        print("Loading weights and getting completions with VLLM")
        load_hf_params_to_vllm(new_params, vllm_model.llm)
        
        if moe_mode:
            rewards = np.zeros(len(batch_ix))
            res = TaskResult(aggregate_metrics={}, sample_details=[None] * len(batch_ix))
            batch_cnt_start = 0
            batch_cnt_end = 0
            for task_name, train_eval_i in train_eval.items():
                batch_cnt_end += len(train_eval_i.samples)
                
                task_mask = (np.array(batch_ix) >= batch_cnt_start) & (np.array(batch_ix) < batch_cnt_end)
                task_batch_ix = [batch_ix_i - batch_cnt_start for batch_ix_i in batch_ix if batch_ix_i>=batch_cnt_start and batch_ix_i<batch_cnt_end]
                # assert len(task_batch_ix) == task_mask.sum()
                res_i = eval_model(vllm_model, train_eval_i, task_batch_ix)
                for task_batch_ix_j, res_i_j in zip(task_batch_ix, res_i.sample_details):
                    list_idx = batch_ix.tolist().index(task_batch_ix_j + batch_cnt_start)
                    res.sample_details[list_idx] = res_i_j
                res.aggregate_metrics.update(res_i.aggregate_metrics)
                rewards[task_mask] = self.get_rewards(task_loader=task_loader, res=res_i)
                
                batch_cnt_start = batch_cnt_end
            # for res_i_j in res.sample_details:
            #     assert res_i_j is not None
        else:
            res = eval_model(vllm_model, train_eval, batch_ix)
            rewards = self.get_rewards(task_loader=task_loader, res=res)
            
        rw_stats = get_mean_std_max_min_dict(array=rewards, prefix="rewards")
        metrics_to_log.update(**rw_stats)

        # 4. 恢复base，并计算ref_log_probs
        if use_kl_loss:
            with torch.no_grad():
                load_base_params(model=model, base_params=original_model_params)
                ref_log_probs_list = self.compute_ref_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    res=res,
                )
                # 5. 加载当前参数
                new_params = forward(
                    policy, model, base_params, decomposed_params, learnable_params
                )

        # 6. 计算policy gradient
        print("Computing the policy gradient...")
        for j, prompt in enumerate(tqdm(prompts)):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + res.sample_details[j]["output"],
                return_tensors="pt",
            ).input_ids.to(gpu)
            generated_ids = output_ids[:, prompt_length:]

            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2, generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            log_likelihood = selected_log_probs.sum(axis=-1)

            pg = -log_likelihood * rewards[j]
            loss = pg

            if use_kl_loss:
                ref_log_probs = ref_log_probs_list[j].to(gpu)
                kl_div = F.kl_div(
                    input=log_probs,
                    target=ref_log_probs,
                    log_target=True,
                    reduction="sum",
                )
                loss = loss + kl_ref_coeff * kl_div
            scaled_loss = loss / clipped_batch_size
            scaled_loss.backward()
            log_dict = {
                "pg": pg.item(),
                "loss": loss.item(),
            }
            if use_kl_loss:
                log_dict["kl_div"] = kl_div.item()
            metrics_to_log.update(**log_dict)
        backward(policy, model, base_params, decomposed_params, learnable_params)

    def update(self, policy):
        max_grad_norm = self.max_grad_norm
        torch.nn.utils.clip_grad_norm_(policy.trainable_params, max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def log_optim(self, metrics_to_log):
        metrics_dict = metrics_to_log.get()
        pg = metrics_dict["pg"]
        print(f"PG={pg}")
        if self.use_kl_loss:
            kl_div = metrics_dict["kl_div"]
            print(f"kl_div={kl_div}")


class RandomShooting(OptimizationAlgorithm, nn.Module):
    def __init__(
        self,
        policy,
        gpu,
        pop_size,
        min_trainable_param,
        max_trainable_param,
        optim_ema=0,
        re_eval_best=True,
        use_loglikelihood_for_ties=False,
        **kwargs,
    ):

        nn.Module.__init__(self=self)
        self.gpu = gpu
        trainable_params = policy.trainable_params
        self.pop_size = pop_size
        self.min_trainable_param = min_trainable_param
        self.max_trainable_param = max_trainable_param
        self.range_trainable_param = max_trainable_param - min_trainable_param
        assert optim_ema >= 0 and optim_ema < 1
        self.optim_ema = optim_ema
        self.re_eval_best = re_eval_best
        self.use_loglikelihood_for_ties = use_loglikelihood_for_ties
        
        """
        pop_size: 这是种群的大小，表示在每次优化步骤中将生成的候选解的数量。较大的种群可以提供更多的多样性，但也会增加计算开销。
        min_trainable_param: 这是可训练参数的最小值，用于限制生成的参数值的下界。确保生成的参数不会低于此值。
        max_trainable_param: 这是可训练参数的最大值，用于限制生成的参数值的上界。确保生成的参数不会超过此值。
        optim_ema: 这是优化的指数移动平均（Exponential Moving Average）系数，范围在0到1之间。它用于平滑参数更新，帮助在训练过程中保持稳定性。
        re_eval_best: 这是一个布尔值，
        use_loglikelihood_for_ties: 这是一个布尔值，指示在处理多个具有相同性能的候选解时，是否使用对数似然值来决定最佳解。如果为True，则会考虑对数似然值来打破平局
        """
        
        self.trainable_params_shapes = [p.shape for p in trainable_params]
        self.trainable_params_nums = [torch.numel(p) for p in trainable_params]
        self.trainable_params_dtype = trainable_params[0].dtype
        self.total_trainable_params = sum(self.trainable_params_nums)
        self.best_idx = 0

        # 定义种群初始值
        initial_values = (
            torch.rand(size=[pop_size, self.total_trainable_params])
            * self.range_trainable_param
        ) + self.min_trainable_param # shape (pop_size, 3*num_layers)
        init_values_flat = [
            torch.flatten(torch.detach_copy(p.data)) for p in trainable_params
        ]
        init_soln = torch.concat(init_values_flat, dim=0) # shape (3*num_layers,)
        
        # 如果需要重新评估最佳解，则将初始值设置为初始解
        if self.re_eval_best:
            initial_values[0] = torch.clone(init_soln)

        self.pop_params = nn.Parameter(
            initial_values,
            requires_grad=False,
        ).cpu() # shape (pop_size, 3*num_layers)
        self.best_soln = nn.Parameter(init_soln, requires_grad=False).cpu() # shape (3*num_layers,)

    def compute_logprobs(
        self,
        model,
        tokenizer,
        prompts,
        generated_outputs,
    ):
        selected_log_probs_list = []
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + generated_outputs[j],
                return_tensors="pt",
            ).input_ids.to(self.gpu)
            generated_ids = output_ids[:, prompt_length:]

            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2, generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            selected_log_probs_list.append(selected_log_probs.detach().cpu())
        return selected_log_probs_list

    @torch.no_grad
    def sample_new_params(
        self,
    ):
        pop_values = (
            torch.rand(size=[self.pop_size, self.total_trainable_params])
            * self.range_trainable_param
        ) + self.min_trainable_param
        if self.re_eval_best:
            pop_values[0] = torch.detach_copy(self.best_soln)

        self.pop_params.data.copy_(pop_values)

    def split_and_convert(self, flat_params):
        split_flat_params = torch.split_with_sizes(
            flat_params, split_sizes=self.trainable_params_nums
        )
        split_params = [
            torch.reshape(p, shape=s).to(dtype=self.trainable_params_dtype).to(self.gpu)
            for p, s in zip(split_flat_params, self.trainable_params_shapes)
        ]
        return split_params

    def get_params_for_pop_member(self, pop_idx):
        return self.split_and_convert(self.pop_params[pop_idx])

    @torch.no_grad
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        self.sample_new_params()
        perf_per_pop = []
        avg_log_likelihoods_per_pop = []
        
        # 计算种群中每一个个体的性能，并计算正确回答的log_prob
        for pop_idx in range(self.pop_size):
            pop_idx_params = self.split_and_convert(
                flat_params=self.pop_params[pop_idx]
            )
            policy.set_trainable_params_values(new_values=pop_idx_params)
            learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )

            print("Loading weights and getting completions with VLLM")
            load_hf_params_to_vllm(new_params, vllm_model.llm)
            res = eval_model(vllm_model, train_eval, batch_ix)
            if self.use_loglikelihood_for_ties:
                print("Storing log likelihhods")
                rewards = task_loader.get_rewards(res=res)
                correct = [int(r > 0) for r in rewards]
                correct_batch_ix = [i for i, c in zip(batch_ix, correct) if c]
                if len(correct_batch_ix) > 0:
                    avg_log_likelihoods = []
                    correct_prompts = [
                        task_loader.get_prompt(
                            tokenizer,
                            train_data,
                            i,
                            model_id=model_id,
                        )
                        for i in correct_batch_ix
                    ]
                    correct_outputs = [
                        res.sample_details[j]["output"]
                        for j, c in enumerate(correct)
                        if c
                    ]
                    selected_log_probs_list = self.compute_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=correct_prompts,
                        generated_outputs=correct_outputs,
                    )
                    for selected_log_probs in selected_log_probs_list:
                        avg_log_likelihood = selected_log_probs.mean(axis=-1)
                        avg_log_likelihoods.append(avg_log_likelihood.item())
                    avg_log_likelihoods_per_pop.append(np.mean(avg_log_likelihoods))
                else:
                    avg_log_likelihoods_per_pop.append(0.0)

            perf = res.aggregate_metrics[task_loader.target_metric_train]
            perf_per_pop.append(perf)

        perf_stats = get_mean_std_max_min_dict(array=perf_per_pop, prefix="pop_perf")
        metrics_to_log.update(**perf_stats)

        if self.use_loglikelihood_for_ties:
            perf_per_pop_array = np.array(perf_per_pop)
            loglikelihood_array = np.array(avg_log_likelihoods_per_pop)
            max_perf = perf_per_pop_array == np.max(perf_per_pop_array)
            max_perf_idxs = np.flatnonzero(max_perf)
            max_perf_logprobs = loglikelihood_array[max_perf_idxs]
            print("SC CHECK")
            print(perf_per_pop)
            print(loglikelihood_array)
            print(max_perf_idxs)
            best_logprob_idx = np.argmax(max_perf_logprobs)
            best_member_idx = max_perf_idxs[best_logprob_idx]
            print(best_logprob_idx)
            print(best_member_idx)
            logprobs_stats = get_mean_std_max_min_dict(
                array=max_perf_logprobs, prefix="logprobs_correct"
            )
            metrics_to_log.update(**logprobs_stats)
        else:
            best_member_idx = np.argmax(perf_per_pop)
        self.best_idx = best_member_idx
        best_params = self.pop_params[best_member_idx].cpu()
        self.best_soln.data.copy_(
            best_params * (1 - self.optim_ema) + self.optim_ema * self.best_soln.cpu()
        )

    def update(self, policy):
        policy.set_trainable_params_values(
            new_values=self.split_and_convert(self.best_soln)
        )

    def log_optim(self, metrics_to_log):
        pass


class CEM(RandomShooting):
    def __init__(
        self,
        policy,
        gpu,
        elite_ratio,
        pop_size,
        min_trainable_param,
        max_trainable_param,
        optim_ema=0,
        re_eval_best=True,
        use_loglikelihood_for_ties=False,
        **kwargs,
    ):

        RandomShooting.__init__(
            self=self,
            policy=policy,
            gpu=gpu,
            pop_size=pop_size,
            min_trainable_param=min_trainable_param,
            max_trainable_param=max_trainable_param,
            optim_ema=optim_ema,
            re_eval_best=re_eval_best,
            use_loglikelihood_for_ties=use_loglikelihood_for_ties,
            **kwargs,
        )

        self.elite_ratio = elite_ratio
        self.num_elites = int(elite_ratio * pop_size)
        """
        self.elite_ratio: 这个参数表示在每次优化步骤中，种群中被认为是“精英”的个体所占的比例。精英个体是指在当前种群中表现最好的个体。通过设置这个比例，可以控制在每次迭代中保留多少表现优秀的个体，以便在下一次迭代中使用它们的参数。
        """
        
        # 定义均值和方差
        self.dist_mean = nn.Parameter(
            torch.detach_copy(self.best_soln), requires_grad=False
        ).cpu()
        init_stdev = (
            torch.ones([self.total_trainable_params]) * self.range_trainable_param / 2
        )
        self.dist_std = nn.Parameter(init_stdev, requires_grad=False).cpu()

    @torch.no_grad
    def sample_new_params(
        self,
    ):
        pop_values = (
            torch.randn(size=[self.pop_size, self.total_trainable_params])
            * self.dist_std
        ) + self.dist_mean
        pop_values = torch.clamp(
            pop_values,
            min=self.min_trainable_param,
            max=self.max_trainable_param,
        )
        if self.re_eval_best:
            pop_values[0] = torch.detach_copy(self.best_soln)

        self.pop_params.data.copy_(pop_values)

    @torch.no_grad
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        self.sample_new_params()
        perf_per_pop = []
        avg_log_likelihoods_per_pop = []
        for pop_idx in range(self.pop_size):
            pop_idx_params = self.split_and_convert(
                flat_params=self.pop_params[pop_idx]
            )
            policy.set_trainable_params_values(new_values=pop_idx_params)
            learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )

            print("Loading weights and getting completions with VLLM")
            load_hf_params_to_vllm(new_params, vllm_model.llm)
            res = eval_model(vllm_model, train_eval, batch_ix)
            if self.use_loglikelihood_for_ties:
                # 计算正确回答的log_prob
                print("Storing log likelihhods")
                rewards = task_loader.get_rewards(res=res)
                correct = [int(r > 0) for r in rewards]
                correct_batch_ix = [i for i, c in zip(batch_ix, correct) if c]
                if len(correct_batch_ix) > 0:
                    avg_log_likelihoods = []
                    correct_prompts = [
                        task_loader.get_prompt(
                            tokenizer,
                            train_data,
                            i,
                            model_id=model_id,
                        )
                        for i in correct_batch_ix
                    ]
                    correct_outputs = [
                        res.sample_details[j]["output"]
                        for j, c in enumerate(correct)
                        if c
                    ]
                    print(f"[pop_idx {pop_idx}] lalala, I am hitting the selected_log_probs!")
                    selected_log_probs_list = self.compute_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=correct_prompts,
                        generated_outputs=correct_outputs,
                    )
                    for selected_log_probs in selected_log_probs_list:
                        avg_log_likelihood = selected_log_probs.mean(axis=-1)
                        avg_log_likelihoods.append(avg_log_likelihood.item())
                    avg_log_likelihoods_per_pop.append(np.mean(avg_log_likelihoods))
                else:
                    avg_log_likelihoods_per_pop.append(0.0)

            perf = res.aggregate_metrics[task_loader.target_metric_train]
            perf_per_pop.append(perf)

        perf_stats = get_mean_std_max_min_dict(array=perf_per_pop, prefix="pop_perf")
        metrics_to_log.update(**perf_stats)

        if self.use_loglikelihood_for_ties:
            perf_per_pop_array = np.array(perf_per_pop)
            loglikelihood_array = np.array(avg_log_likelihoods_per_pop)
            # 找出所有性能最高的个体
            max_perf = perf_per_pop_array == np.max(perf_per_pop_array)
            max_perf_idxs = np.flatnonzero(max_perf)
            # 找出所有性能最高的个体中，正确回答的log_prob
            max_perf_logprobs = loglikelihood_array[max_perf_idxs]
            # 找出所有性能最高的个体中，正确回答的log_prob最大的个体
            best_logprob_idx = np.argmax(max_perf_logprobs)
            best_member_idx = max_perf_idxs[best_logprob_idx]
            logprobs_stats = get_mean_std_max_min_dict(
                array=max_perf_logprobs, prefix="logprobs_correct"
            )
            metrics_to_log.update(**logprobs_stats)
        else:
            best_member_idx = np.argmax(perf_per_pop)
            
        # 选择精英个体
        elite_idxs = np.argpartition(perf_per_pop, -self.num_elites)[-self.num_elites :] # 返回的是精英个体的索引

        elite_params = self.pop_params[elite_idxs]
        elite_mean = torch.mean(elite_params, dim=0)
        elite_std = torch.std(elite_params, dim=0)
        self.best_idx = best_member_idx
        best_params = self.pop_params[best_member_idx].cpu()
        
        # 更新最佳种群，以及精英种群的均值和方差
        self.best_soln.data.copy_(best_params)
        self.dist_mean.copy_(
            elite_mean.cpu() * (1 - self.optim_ema)
            + self.optim_ema * self.dist_mean.cpu()
        )
        self.dist_std.copy_(
            elite_std.cpu() * (1 - self.optim_ema)
            + self.optim_ema * self.dist_std.cpu()
        )

        cem_mean_stats = get_mean_std_max_min_dict(
            array=self.dist_mean.detach().cpu().numpy(),
            prefix="cem_mean",
        )
        metrics_to_log.update(**cem_mean_stats)

        cem_std_stats = get_mean_std_max_min_dict(
            array=self.dist_std.detach().cpu().numpy(),
            prefix="cem_std",
        )
        metrics_to_log.update(**cem_std_stats)



import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass
class AttackConfig:
    """Configuration describing the textual layout of the attack."""

    pre_text: str
    post_text: str
    target_text: str
    suffix_length: int
    log_every: int = 20


class PrimaryModelWrapper:
    """Wraps a causal language model used as the optimization backbone."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: AttackConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        
        self.pre_ids = self._encode(config.pre_text)
        self.post_ids = self._encode(config.post_text)
        self.target_ids = self._encode(config.target_text)

        self.model.eval()

    def _encode(self, text: str) -> torch.LongTensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, device=self.device, dtype=torch.long)

    def suffix_string(self, suffix_ids: torch.LongTensor) -> str:
        return self.tokenizer.decode(
            suffix_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def decode_prompt(self, suffix_ids: torch.LongTensor) -> str:
        return (
            self.tokenizer.decode(self.pre_ids.tolist(), clean_up_tokenization_spaces=False)
            + self.suffix_string(suffix_ids)
            + self.tokenizer.decode(self.post_ids.tolist(), clean_up_tokenization_spaces=False)
        )

    def _build_full_ids(self, suffix_ids: torch.LongTensor) -> torch.LongTensor:
        return torch.cat([self.pre_ids, suffix_ids, self.post_ids, self.target_ids])

    def _build_labels(self, suffix_ids: torch.LongTensor) -> torch.LongTensor:
        ignore_len = self.pre_ids.numel() + suffix_ids.numel() + self.post_ids.numel()
        ignore = torch.full((ignore_len,), -100, device=self.device, dtype=torch.long)
        return torch.cat([ignore, self.target_ids])

    def compute_loss(self, suffix_ids: torch.LongTensor) -> float:
        with torch.no_grad():
            input_ids = self._build_full_ids(suffix_ids)
            attention_mask = torch.ones_like(input_ids)
            labels = self._build_labels(suffix_ids)
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                labels=labels.unsqueeze(0),
                return_dict=True,
            )
            return float(outputs.loss.item())

    def gradient(self, suffix_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, float]:
        input_ids = self._build_full_ids(suffix_ids)
        attention_mask = torch.ones_like(input_ids)
        labels = self._build_labels(suffix_ids)

        embed_layer: nn.Embedding = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        inputs_embeds = inputs_embeds.clone().detach()
        inputs_embeds.requires_grad_(True)

        outputs = self.model(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            labels=labels.unsqueeze(0),
            return_dict=True,
        )
        loss = outputs.loss
        loss.backward()

        grads = inputs_embeds.grad.detach()
        suffix_start = self.pre_ids.numel()
        suffix_end = suffix_start + suffix_ids.numel()
        suffix_grads = grads[suffix_start:suffix_end]

        self.model.zero_grad(set_to_none=True)
        return suffix_grads, float(loss.item())

    def generate(self, suffix_ids: torch.LongTensor, max_new_tokens: int = 200) -> str:
        prompt = self.decode_prompt(suffix_ids)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        output_ids = generated[0][encoded["input_ids"].shape[-1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


class AuxiliaryModelWrapper:
    """Wraps additional models used to evaluate transferability."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: AttackConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model.eval()

    def compute_loss(self, suffix_text: str) -> float:
        prompt_text = f"{self.config.pre_text}{suffix_text}{self.config.post_text}"
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(self.config.target_text, add_special_tokens=False)
        input_tensor = torch.tensor(input_ids + target_ids, device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)

        ignore_prefix = torch.full((len(input_ids),), -100, device=self.device, dtype=torch.long)
        labels = torch.cat([ignore_prefix, torch.tensor(target_ids, device=self.device, dtype=torch.long)])

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                labels=labels.unsqueeze(0),
                return_dict=True,
            )
        return float(outputs.loss.item())

    def generate(self, suffix_text: str, max_new_tokens: int = 200) -> str:
        prompt_text = f"{self.config.pre_text}{suffix_text}{self.config.post_text}"
        encoded = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        output_ids = generated[0][encoded["input_ids"].shape[-1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


class GCGOptimizer:
    """Implements a greedy coordinate gradient optimizer for suffix tokens."""

    def __init__(
        self,
        primary: PrimaryModelWrapper,
        aux_models: Optional[Sequence[AuxiliaryModelWrapper]] = None,
        top_k: int = 40,
        random_seed: int = 0,
    ) -> None:
        self.primary = primary
        self.aux_models = list(aux_models) if aux_models else []
        self.top_k = top_k
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        embed_layer: nn.Embedding = self.primary.model.get_input_embeddings()
        self.embedding_weight = embed_layer.weight.detach()

    def _initial_suffix(self, vocab_size: int, device: torch.device, length: int) -> torch.LongTensor:
        
        space_id = None
        try:
            space_id = self.primary.tokenizer.encode(" ", add_special_tokens=False)[0]
        except Exception:
            space_id = None
        if space_id is not None:
            ids = torch.full((length,), space_id, device=device, dtype=torch.long)
        else:
            ids = torch.randint(0, vocab_size, (length,), device=device)
        return ids

    def optimize(
        self,
        num_iterations: int,
        success_callback: Optional[Callable[[torch.LongTensor], bool]] = None,
        verbose: bool = True,
    ) -> Tuple[torch.LongTensor, List[Tuple[int, float]], int, Dict[int, bool]]:
        config = self.primary.config
        suffix_ids = self._initial_suffix(
            vocab_size=self.embedding_weight.size(0),
            device=self.primary.device,
            length=config.suffix_length,
        )

        history: List[Tuple[int, float]] = []

        success_history: Dict[int, bool] = {}
        current_loss = self.primary.compute_loss(suffix_ids)
        first_success_iter = -1

        for iteration in range(1, num_iterations + 1):
            grads, loss_val = self.primary.gradient(suffix_ids)
            current_loss = loss_val

            improved = False
            for position in range(config.suffix_length):
                grad_vec = grads[position]
                if torch.allclose(grad_vec, torch.zeros_like(grad_vec)):
                    continue

                scores = torch.matmul(self.embedding_weight, (-grad_vec))
                _, candidate_ids = torch.topk(scores, k=self.top_k)

                best_candidate = suffix_ids[position].item()
                best_loss = current_loss

                for token_id in candidate_ids.tolist():
                    if token_id == suffix_ids[position].item():
                        continue
                    candidate_suffix = suffix_ids.clone()
                    candidate_suffix[position] = token_id

                    candidate_loss = self.primary.compute_loss(candidate_suffix)
                    if self.aux_models:
                        suffix_text = self.primary.suffix_string(candidate_suffix)
                        aux_loss = sum(aux.compute_loss(suffix_text) for aux in self.aux_models)
                        candidate_loss = (candidate_loss + aux_loss) / (len(self.aux_models) + 1)

                    if candidate_loss + 1e-6 < best_loss:
                        best_loss = candidate_loss
                        best_candidate = token_id

                if best_candidate != suffix_ids[position].item():
                    suffix_ids[position] = best_candidate
                    current_loss = best_loss
                    improved = True

            if iteration % config.log_every == 0 or iteration == 1 or iteration == num_iterations:
                history.append((iteration, current_loss))
                if verbose:
                    prompt_text = self.primary.decode_prompt(suffix_ids)
                    print(f"Iter {iteration:04d} | Loss: {current_loss:.4f} | Prompt tail: ...{prompt_text[-60:]}")
                
                # MODIFIED: Check and record success at this specific iteration
                if success_callback:
                    is_success = success_callback(suffix_ids)
                    success_history[iteration] = is_success # NEW: Record success status
                    if is_success and first_success_iter == -1:
                        first_success_iter = iteration
                        if verbose:
                            print(f"--- Success found at iteration {iteration}! ---")

            if not improved and iteration > 1:
                if verbose:
                    print(f"No improvement at iteration {iteration}, stopping early.")
                break
        
        # MODIFIED: Return the new success_history dictionary
        return suffix_ids, history, first_success_iter, success_history


def save_history(history: Iterable[Tuple[int, float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("iteration,loss\n")
        for iteration, loss in history:
            f.write(f"{iteration},{loss}\n")
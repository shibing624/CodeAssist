# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Rewrite the original WizardCoder to support code completion.
"""

import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Dict, Sequence, Union

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
has_cuda = torch.cuda.is_available()
IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class WizardCoder:
    def __init__(
            self,
            model_name_or_path: str = "WizardLM/WizardCoder-15B-V1.0",
            peft_name: Optional[str] = None,
            special_words_dict: Dict = None,
            use_cuda: Optional[bool] = has_cuda,
            cuda_device: Optional[int] = -1,
            fp16: bool = True,
            bf16: bool = False,
            **kwargs,
    ):
        """
        Initializes a AutoModelForCausalLM
    
        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            peft_name: The name of the PEFT model to use.
            special_words_dict: A dictionary of special words and their token ids.
            use_cuda: Use GPU if available.
            cuda_device: Which cuda device to use.
            fp16: Use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.
            bf16: Use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.
            **kwargs: Additional kwargs for the Transformers `PreTrainedModel` and `PreTrainedTokenizer` classes.
        """
        self.model_name_or_path = model_name_or_path
        self.fp16 = fp16
        self.bf16 = bf16
        self.device_map = "auto"
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
                    self.device_map = {"": int(cuda_device)}
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_map = {"": "mps"}
            else:
                self.device = "cpu"
                self.device_map = {"": "cpu"}
        logger.debug(f"Device: {self.device}")
        if not use_cuda:
            self.fp16 = False
            self.bf16 = False
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = world_size != 1
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        if torch.cuda.is_bf16_supported() and not self.bf16:
            logger.warning("GPU supports bf16, you can enable bf16.")
        self.torch_dtype = torch.bfloat16 if self.bf16 else (torch.float16 if self.fp16 else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            **kwargs,
        )
        if peft_name:
            # Load PEFT model for inference default, if you want to continue training, please set is_trainable=True
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                is_trainable=False,
            )
            logger.info(f"Loaded peft model from {peft_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # Set padding side equal to Collator padding side
        self.tokenizer.padding_side = "left"

        if "starcoder" in model_name_or_path:
            self.tokenizer.add_special_tokens({
                "eos_token": "<|endoftext|>",
                "bos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            })
        if special_words_dict is not None:
            self.tokenizer.add_special_tokens(special_words_dict)
        self.results = {}

    def set_seed(self, seed):
        logger.debug(f"Set seed for random, numpy and torch: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_model(
            self,
            train_file: str,
            output_dir: str,
            eval_file: str = None,
            batch_size: int = 8,
            num_epochs: int = 1,
            lr: float = 5e-5,
            gradient_accumulation_steps: int = 1,
            max_steps: int = -1,
            logging_steps: int = 50,
            gradient_checkpointing: bool = True,
            torch_compile: bool = False,
            warmup_steps: int = 200,
            save_steps: int = 400,
            eval_steps: int = 200,
            optimizer: str = "adamw_torch",
            save_strategy: str = "steps",
            save_total_limit: int = 10,
            report_to: Optional[List[str]] = "tensorboard",
            overwrite_output_dir: bool = True,
            use_peft: bool = True,
            int8: bool = False,
            max_eval_samples: int = 20,
            **kwargs,
    ):
        """
        Trains the model on 'train_file'

        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            eval_file (optional): Path to eval file containing the text to evaluate the language model on.
            batch_size (optional): Batch size for training.
            num_epochs (optional): Number of epochs for training.
            lr (optional): Learning rate.
            gradient_accumulation_steps (optional): Number of updates steps to accumulate before performing a backward/update pass.
            max_steps (optional): If > 0: set total number of training steps to perform. Override num_epochs.
            logging_steps (optional): Number of steps between logging.
            gradient_checkpointing (optional): If True, use gradient checkpointing to save memory at the expense of slower backward pass.
            torch_compile (optional): If True, use torch's new experimental just-in-time (JIT) compiler to compile the model for faster runtime.
            warmup_steps (optional): Number of steps for the warmup in the lr scheduler.
            save_steps (optional): Number of steps between saving.
            eval_steps (optional): Number of steps between evaluations.
            optimizer (optional): Optimizer to use. Can be 'adamw_torch', 'adamw_deepspeed', 'adam', 'sgd', 'lamb' or 'lamb_wd'.
            save_strategy (optional): Strategy to save checkpoints. Can be 'steps' or 'epoch'.
            save_total_limit (optional): Maximum number of checkpoints to keep.
            report_to (optional): The list of integrations to report the results and logs to.
            overwrite_output_dir (optional): Overwrite the content of the output directory.
            use_peft (optional): If True, use the PEFT scheduler to schedule the training.
            int8 (optional): If True, use int8 quantization for the model.
            max_eval_samples (optional): Maximum number of samples to use for evaluation.
            kwargs (optional): Optional model specific arguments.

        Returns:
            global_step: Number of global steps trained
            metrics: Dictionary containing the evaluation results.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.set_seed(42)
        logger.debug(f"Tokenizer: {self.tokenizer}")
        logger.debug(f"Model: {self.model}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            dataloader_drop_last=True,
            learning_rate=lr,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_checkpointing=gradient_checkpointing,
            torch_compile=torch_compile,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            optim=optimizer,
            save_strategy=save_strategy,
            evaluation_strategy='steps' if eval_file is not None else 'no',
            eval_steps=eval_steps if eval_file is not None else None,
            load_best_model_at_end=True if eval_file is not None else False,
            ddp_find_unused_parameters=False,
            save_total_limit=save_total_limit,
            fp16=self.fp16,
            bf16=self.bf16,
            report_to=report_to,
            overwrite_output_dir=overwrite_output_dir,
            no_cuda=True if self.device == "cpu" else False,
            **kwargs
        )
        # update model train config
        if training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.config.use_cache = True
        self.model.enable_input_require_grads()

        # Tell Trainer not to attempt DataParallel
        self.model.is_parallelizable = True
        self.model.model_parallel = True
        # Setup peft
        if use_peft:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["c_proj", "c_attn", "q_attn"],
                bias="none",
            )
            if int8:
                self.model = prepare_model_for_int8_training(self.model)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        else:
            logger.warning("Now full model params fine-tune, which is slow, set `use_peft=True` for lora fine-tune.")
        logger.debug(f"Tokenizer: {self.tokenizer}")
        logger.debug(f"Model: {self.model}")

        # load dataset
        raw_train_datasets = load_dataset('json', data_files=train_file, split="train")
        logger.debug(f"Example train_dataset[0]: {raw_train_datasets[0]}")
        with training_args.main_process_first(desc="Train dataset tokenization"):
            train_dataset = raw_train_datasets.map(
                self.train_tokenize_function,
                batched=True,
                num_proc=1,
                remove_columns=raw_train_datasets.column_names,
                desc="Running tokenizer on train dataset",
                fn_kwargs={"tokenizer": self.tokenizer}
            )
            logger.debug(f"Train dataset size: {len(train_dataset)}")
            logger.debug(f"First sample of the training set: {train_dataset[0]}.")
        eval_dataset = None
        if eval_file is not None:
            raw_eval_datasets = load_dataset('json', data_files=eval_file, split="train")
            if max_eval_samples is not None and max_eval_samples > 0:
                max_eval_samples = min(len(raw_eval_datasets), max_eval_samples)
                raw_eval_datasets = raw_eval_datasets.select(range(max_eval_samples))
            with training_args.main_process_first(desc="Eval dataset tokenization"):
                eval_dataset = raw_eval_datasets.map(
                    self.train_tokenize_function,
                    batched=True,
                    num_proc=1,
                    remove_columns=raw_eval_datasets.column_names,
                    desc="Running tokenizer on train dataset",
                    fn_kwargs={"tokenizer": self.tokenizer}
                )
                logger.debug(f"Eval dataset size: {len(eval_dataset)}")
                logger.debug(f"First sample of the eval set: {eval_dataset[0]}.")

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        if training_args.local_rank <= 0:
            logger.info(f"Training/evaluation parameters {training_args}")

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        logger.info("*** Train ***")
        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")
        (global_step, training_loss, metrics) = trainer.train()
        self.results.update(metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        self.save_model(output_dir=output_dir)
        logger.info(f" Training model done. Saved to {output_dir}.")

        if eval_dataset is not None:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            metrics['eval_samples'] = len(eval_dataset)
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
            logger.debug(f"eval metrics: {metrics}")
            self.results.update(metrics)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if training_args.local_rank <= 0:
            logger.debug(f"metrics: {self.results}")
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.model_name_or_path, output_dir
                )
            )

        return global_step, metrics

    def save_model(self, output_dir):
        """Save the model and the tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        model = self.model
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def smart_tokenizer_and_embedding_resize(
            self,
            special_tokens_dict: Dict,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            input_ids_lens=input_ids_lens,
        )

    def preprocess(
            self,
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer: PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def train_tokenize_function(self, examples, tokenizer):
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        if 'input' in examples:
            sources = [
                prompt_input.format_map(dict(instruction=instruction, input=input)) if input != ""
                else prompt_no_input.format_map(dict(instruction=instruction))
                for instruction, input in zip(examples['instruction'], examples['input'])
            ]
        else:
            sources = [
                prompt_no_input.format_map(dict(instruction=instruction))
                for instruction in examples['instruction']
            ]
        targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
        data_dict = self.preprocess(sources, targets, tokenizer)
        return data_dict

    @torch.inference_mode()
    def generate(
            self,
            sentences: Union[List[str], str],
            keep_prompt: bool = False,
            add_system_prompt: bool = True,
            eval_batch_size: int = 4,
            max_length: int = 256,
            temperature: int = 1.0,
            top_k: int = 50,
            top_p: float = 0.95,
            do_sample: bool = True,
            num_beams: int = 1,
            repetition_penalty: float = 1.0,
            length_penalty: float = 1.2,
            early_stopping: bool = True,
            verbose: bool = True,
            **kwargs
    ) -> List[str]:
        """
        Performs predictions on a list of text.

        Args:
            sentences: A prompt text for the model.
            keep_prompt: Whether to keep the prompt in the generated text.
            add_system_prompt: Whether to add the system prompt to the prompt text.
            eval_batch_size: The batch size for evaluation.
            max_length: The maximum length of the generated text.
            temperature: The sampling temperature.
            top_k: The number of top k tokens to be considered by sampling.
            top_p: The sampling probability for top p tokens.
            num_beams: The number of beams for beam search.
            repetition_penalty: The repetition penalty parameter.
            do_sample: Boolean value indicating whether to sample or greedy generate.
            length_penalty: The length penalty parameter.
            early_stopping: Boolean value indicating whether to do early stopping or not.
            verbose: Boolean value indicating whether to print the progress bar.
            **kwargs: Additional arguments for the generate method of the model.
        Returns:
            generated_sequences: list, Sequences of text generated by the model.
        """
        all_outputs = []
        if isinstance(sentences, str):
            sentences = [sentences]
        for batch in tqdm(
                [sentences[i: i + eval_batch_size] for i in range(0, len(sentences), eval_batch_size)],
                desc="Generating outputs",
                disable=not verbose,
        ):
            if add_system_prompt:
                batch = [PROMPT_DICT['prompt_no_input'].format(instruction=s) for s in batch]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            outputs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
            for idx, (prompt_text, generated_sequence) in enumerate(zip(batch, outputs.sequences)):
                # Decode text
                text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                prompt_len = len(prompt_text)
                gen_text = text[prompt_len:]
                if keep_prompt:
                    total_sequence = prompt_text + gen_text
                else:
                    total_sequence = gen_text
                all_outputs.append(total_sequence)
        return all_outputs


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

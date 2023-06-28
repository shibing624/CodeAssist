# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Rewrite the original gpt2 model to support the autocomplete
"""

import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Sequence

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            model_name_or_path: str,
            max_seq_length: int = 512,
            do_lower_case: bool = False,
            special_words_dict: Dict = None
    ):
        """
        Initializes a GPT2 LanguageModelingModel.
    
        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            max_seq_length: The maximum total input sequence length after tokenization.
            do_lower_case: Set this flag if you are using an uncased model.
            special_words_dict: A dictionary of special words and their token ids.
        """
        self.model_name_or_path = model_name_or_path
        self.do_lower_case = do_lower_case
        if max_seq_length > 1024:
            logger.warning("GPT only allows a max_seq_length of 1024. Value will be set to 1024")
            max_seq_length = 1024
        self.max_seq_length = max_seq_length
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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
            fp16: bool = True,
            bf16: bool = False,
            report_to: Optional[List[str]] = "tensorboard",
            overwrite_output_dir: bool = True,
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
            fp16 (optional): Set to True to use apex for mixed precision training.
            bf16 (optional): Set to True to use deepspeed BF16 precision.
            report_to (optional): The list of integrations to report the results and logs to.
            overwrite_output_dir (optional): Overwrite the content of the output directory.
            kwargs (optional): Optional model specific arguments.

        Returns:
            global_step: Number of global steps trained
            metrics: Dictionary containing the evaluation results.
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.debug(f"Tokenizer: {self.tokenizer}")
        logger.debug(f"Model: {self.model}")

        # load dataset
        raw_train_datasets = load_dataset('json', data_files=train_file, split="train")

        train_dataset = raw_train_datasets.map(
            self.train_tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=raw_train_datasets.column_names,
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": self.tokenizer}
        )
        logger.debug(f"train dataset size: {len(train_dataset)}")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")
        eval_dataset = None
        if eval_file is not None:
            raw_eval_datasets = load_dataset('json', data_files=eval_file, split="train")

            eval_dataset = raw_eval_datasets.map(
                self.train_tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=raw_eval_datasets.column_names,
                desc="Running tokenizer on train dataset",
                fn_kwargs={"tokenizer": self.tokenizer}
            )

        # start train
        training_args = TrainingArguments(
            output_dir=output_dir,
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
            fp16=fp16,
            bf16=bf16,
            report_to=report_to,
            overwrite_output_dir=overwrite_output_dir,
            no_cuda=True if device == "cpu" else False,
            **kwargs
        )
        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        if training_args.local_rank <= 0:
            logger.info(f"Training/evaluation parameters {training_args}")

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

        # Tell Trainer not to attempt DataParallel
        self.model.is_parallelizable = True
        self.model.model_parallel = True

        trainer = Trainer(model=self.model, tokenizer=self.tokenizer, args=training_args, **data_module)
        self.model.config.use_cache = False

        (global_step, training_loss, metrics) = trainer.train()
        self.results.update(metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        self.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)

        logger.info(f" Training model done. Saved to {output_dir}.")

        if eval_file is not None:
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

    def safe_save_model_for_hf_trainer(self, trainer: Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

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
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
            self,
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer: PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in
                                                 (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = input_ids.copy()
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def train_tokenize_function(self, examples, tokenizer):
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        if 'input' in examples:
            sources = [
                prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
                    else prompt_no_input.format_map(dict(instruction=instruction)) \
                for instruction, input in zip(examples['instruction'], examples['input'])
            ]
        else:
            sources = [
                prompt_no_input.format_map(dict(instruction=instruction)) \
                for instruction in examples['instruction']
            ]
        targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
        data_dict = self.preprocess(sources, targets, tokenizer)
        return data_dict

    def generate(
            self,
            prompt: str,
            is_add_prompt: bool = True,
            temperature: int = 1.0,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            do_sample: bool = True,
            num_return_sequences: int = 1,
            length_penalty: float = 2.0,
            early_stopping: bool = True,
            stop_word: str = "\n\n",
            bad_words: list = None,
            **kwargs
    ):
        """
        Generate text using a GPT2 LanguageGenerationModel

        Args:
            prompt: A prompt text for the model.
            is_add_prompt: Whether to add the prompt to the returned text.
            temperature: The sampling temperature.
            top_k: The number of top k tokens to be considered by sampling.
            top_p: The sampling probability for top p tokens.
            repetition_penalty: The repetition penalty parameter.
            do_sample: Boolean value indicating whether to sample or greedy generate.
            num_return_sequences: The number of samples to return.
            length_penalty: The length penalty parameter.
            early_stopping: Boolean value indicating whether to do early stopping or not.
            stop_word: A stop word to stop generation.
            bad_words: A list of bad words to be ignored.
        Returns:
            generated_sequences: list, Sequences of text generated by the model.
        """
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt").to(device)
        encoded_prompt_ids = encoded_prompt.input_ids
        # Get tokens of words that should not be generated
        bad_words_ids = [self.tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in
                         bad_words] if bad_words else None
        output_sequences = self.model.generate(
            input_ids=encoded_prompt_ids,
            max_length=self.max_seq_length + len(encoded_prompt_ids[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            bad_words_id=bad_words_ids,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,  # tokenizer.pad_token_ids is None
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []
        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after the stop word token
            text = text[: text.find(stop_word) if stop_word else None]
            # Remove the excess text that was used for pre-processing
            total_sequence = text[len(self.tokenizer.decode(encoded_prompt_ids[0], clean_up_tokenization_spaces=True)):]
            # Add the prompt at the beginning of the sequence.
            if is_add_prompt:
                total_sequence = prompt + total_sequence
            generated_sequences.append(total_sequence)

        return generated_sequences


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

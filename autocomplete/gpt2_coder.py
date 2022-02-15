# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Rewrite the original gpt2 model to support the autocomplete
refer: https://github.com/ThilinaRajapakse/simpletransformers/tree/master/simpletransformers/language_modeling
"""

import os
from typing import Dict, List, Union
from loguru import logger
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.data.datasets.language_modeling import TextDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    logger.info(f"Set seed for random, numpy and torch: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GPT2Coder:
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: int = 128,
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
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        # When download from transformers, cache dir: ~/.cache/huggingface/transformers/
        if special_words_dict is not None:
            self.add_special_words(special_words_dict)
        self.results = {}

    def add_special_words(self, special_words_dict):
        origin_num_tokens = len(self.tokenizer)
        num_added_tokens = self.tokenizer.add_special_tokens(special_words_dict)
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(new_num_tokens=origin_num_tokens + num_added_tokens)

    def train_model(
            self,
            train_file: str,
            output_dir: str,
            eval_file: str = None,
            verbose: bool = True,
            batch_size: int = 8,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.1,
            lr: float = 5e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1
    ):
        """
        Trains the model on 'train_file'

        Args:
            train_file: Path to text file containing the text to train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            eval_file (optional): Path to eval file containing the text to evaluate the language model on.
            verbose (optional): Print logger or not.
            batch_size (optional): Batch size for training.
            num_epochs (optional): Number of epochs for training.
            weight_decay (optional): Weight decay for optimization.
            seed (optional): Seed for initialization.
            warmup_ratio (optional): Warmup ratio for learning rate.
            lr (optional): Learning rate.
            eps (optional): Adam epsilon.
            gradient_accumulation_steps (optional): Number of updates steps to accumulate before performing a backward/update pass.
            max_grad_norm (optional): Max gradient norm.
            max_steps (optional): If > 0: set total number of training steps to perform. Override num_epochs.
        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.to(device)
        train_dataset = TextDataset(self.tokenizer, train_file, self.max_seq_length, overwrite_cache=True,
                                    cache_dir=output_dir)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            eval_file=eval_file,
            verbose=verbose,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            seed=seed,
            warmup_ratio=warmup_ratio,
            lr=lr,
            eps=eps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps
        )
        logger.info(f" Training model done. Saved to {output_dir}.")

        return global_step, training_details

    def train(
            self,
            train_dataset: Dataset,
            output_dir: str,
            eval_file: str = None,
            verbose: bool = True,
            batch_size: int = 8,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.1,
            lr: float = 5e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        set_seed(seed)

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate,
        )

        total_steps = len(train_dataloader) * num_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of train data for warm-up
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")

        logger.info("  Training started")
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        epoch_number = 0
        best_eval_metric = 1e3
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to global_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)
                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d" % epochs_trained)
                logger.info("   Continuing training from global step %d" % global_step)
                logger.info("   Will skip the first %d steps in the current epoch" % steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        training_progress_scores = {
            "global_step": [],
            "perplexity": [],
            "eval_loss": [],
            "train_loss": [],
        }
        train_iterator = trange(int(num_epochs), desc="Epoch", disable=False, mininterval=0)
        for current_epoch in train_iterator:
            self.model.train()
            current_loss = 0
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(current_epoch)
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {num_epochs}")
            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Running Epoch {epoch_number} of {num_epochs}",
                                  disable=False,
                                  mininterval=0)
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = batch.to(device)
                outputs = self.model(inputs, labels=inputs)
                loss = outputs[0]
                current_loss = loss.item()
                if verbose:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{num_epochs}. Running Loss: {current_loss:9.4f}")

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
            results = self.eval_model(eval_file, output_dir_current, verbose=verbose, batch_size=batch_size)
            self.save_model(output_dir_current, model=self.model, results=results)
            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])
            report = pd.DataFrame(training_progress_scores)
            report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

            eval_loss = results["eval_loss"]
            if eval_loss < best_eval_metric:
                best_eval_metric = eval_loss
                self.save_model(output_dir, model=self.model, results=results)

            if 0 < max_steps < global_step:
                return global_step, training_progress_scores

        return global_step, training_progress_scores

    def eval_model(self, eval_file: str, output_dir: str = None, verbose: bool = True, batch_size: int = 16):
        """
        Evaluates the model on eval_df. Saves results to args.output_dir
            result: Dictionary containing evaluation results.
        """
        self.model.to(device)
        eval_dataset = TextDataset(self.tokenizer, eval_file, self.max_seq_length, overwrite_cache=True)
        result = self.evaluate(eval_dataset, output_dir, batch_size=batch_size)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result

    def evaluate(self, eval_dataset, output_dir: str = None, batch_size: int = 16):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        results = {}

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=batch_size,
                                     sampler=eval_sampler,
                                     collate_fn=collate)
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()

        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            inputs = batch.to(device)
            with torch.no_grad():
                outputs = self.model(inputs, labels=inputs)
                lm_loss = outputs[0]
                eval_loss += lm_loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        results["eval_loss"] = eval_loss
        results["perplexity"] = perplexity
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def save_model(self, output_dir, model, results=None):
        """
        Saves the model to output_dir.
        :param output_dir:
        :param model:
        :param results:
        :return:
        """
        logger.info("Saving model checkpoint to %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

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

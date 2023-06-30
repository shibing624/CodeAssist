import os
import glob
import io

from .. import data


class IMDB(data.Dataset):

    urls = ['http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
    name = 'imdb'
    dirname = 'aclImdb'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Args:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with io.open(fname, 'r', encoding="utf-8") as f:
                    text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Args:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(IMDB, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the IMDB dataset.

        Args:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)

            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)


from . import raw

__all__ = ['raw']


# *****************************************************************************
# Copyright (c) 2017 Keith Ito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# *****************************************************************************
"""
Modified from https://github.com/keithito/tacotron
"""

import inflect
import re


_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_pounds_re = re.compile(r'([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(text: str) -> str:
    return re.sub(_comma_number_re, lambda m: m.group(1).replace(',', ''), text)


def _expand_pounds(text: str) -> str:
    return re.sub(_pounds_re, r'\1 pounds', text)


def _expand_dollars_repl_fn(m):
    """The replacement function for expanding dollars."""
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    if len(parts) > 1 and parts[1]:
        if len(parts[1]) == 1:
            # handle the case where we have one digit after the decimal point
            cents = int(parts[1]) * 10
        else:
            cents = int(parts[1])
    else:
        cents = 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_dollars(text: str) -> str:
    return re.sub(_dollars_re, _expand_dollars_repl_fn, text)


def _expand_decimal_point(text: str) -> str:
    return re.sub(_decimal_number_re, lambda m: m.group(1).replace('.', ' point '), text)


def _expand_ordinal(text: str) -> str:
    return re.sub(_ordinal_re, lambda m: _inflect.number_to_words(m.group(0)), text)


def _expand_number_repl_fn(m):
    """The replacement function for expanding number."""
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def _expand_number(text: str) -> str:
    return re.sub(_number_re, _expand_number_repl_fn, text)


def normalize_numbers(text: str) -> str:
    text = _remove_commas(text)
    text = _expand_pounds(text)
    text = _expand_dollars(text)
    text = _expand_decimal_point(text)
    text = _expand_ordinal(text)
    text = _expand_number(text)
    return text


import math
import collections
import torch
from torchtext.data.utils import ngrams_iterator


def _compute_ngram_counter(tokens, max_n):
    """ Create a Counter with a count of unique n-grams in the tokens list

    Args:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted

    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count

    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1,
             ('me', 'me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(tuple(x.split(' '))
                                         for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter


def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

    Args:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(references_corpus),\
        'The length of candidate and reference corpus should be the same'

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:  # TODO: no need to loop through the whole counter
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()


import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from torchtext.experimental.datasets import SQuAD1
from model import QuestionAnswerTask
from metrics import compute_qa_exact, compute_qa_f1
from utils import print_loss_log
from model import BertModel, BertEmbedding


def process_raw_data(data):
    _data = []
    for (context, question, answers, ans_pos) in data:
        right_length = True
        for _idx in range(len(ans_pos)):
            if ans_pos[_idx][1] + question.size(0) + 2 >= args.bptt:
                right_length = False
        if right_length:
            _data.append((context, question, answers, ans_pos))
    return _data


def collate_batch(batch):
    seq_list = []
    ans_pos_list = []
    tok_type = []
    for (context, question, answers, ans_pos) in batch:
        qa_item = torch.cat((torch.tensor([cls_id]), question, torch.tensor([sep_id]),
                             context, torch.tensor([sep_id])))
        if qa_item.size(0) > args.bptt:
            qa_item = qa_item[:args.bptt]
        elif qa_item.size(0) < args.bptt:
            qa_item = torch.cat((qa_item,
                                 torch.tensor([pad_id] * (args.bptt -
                                              qa_item.size(0)))))
        seq_list.append(qa_item)
        pos_list = [pos + question.size(0) + 2 for pos in ans_pos]  # 1 for sep and 1 for cls
        ans_pos_list.append(pos_list)
        tok_type.append(torch.cat((torch.zeros((question.size(0) + 2)),
                                   torch.ones((args.bptt -
                                               question.size(0) - 2)))))
    _ans_pos_list = []
    for pos in zip(*ans_pos_list):
        _ans_pos_list.append(torch.stack(list(pos)))
    return torch.stack(seq_list).long().t().contiguous().to(device), \
        _ans_pos_list, \
        torch.stack(tok_type).long().t().contiguous().to(device)


def evaluate(data_source, vocab):
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_batch)
    ans_pred_tokens_samples = []
    with torch.no_grad():
        for idx, (seq_input, ans_pos_list, tok_type) in enumerate(dataloader):
            start_pos, end_pos = model(seq_input, token_type_input=tok_type)
            target_start_pos, target_end_pos = [], []
            for item in ans_pos_list:
                _target_start_pos, _target_end_pos = item.to(device).split(1, dim=-1)
                target_start_pos.append(_target_start_pos.squeeze(-1))
                target_end_pos.append(_target_end_pos.squeeze(-1))
            loss = (criterion(start_pos, target_start_pos[0])
                    + criterion(end_pos, target_end_pos[0])) / 2
            total_loss += loss.item()
            start_pos = nn.functional.softmax(start_pos, dim=1).argmax(1)
            end_pos = nn.functional.softmax(end_pos, dim=1).argmax(1)
            seq_input = seq_input.transpose(0, 1)  # convert from (S, N) to (N, S)
            for num in range(0, seq_input.size(0)):
                if int(start_pos[num]) > int(end_pos[num]):
                    continue  # start pos is in front of end pos
                ans_tokens = []
                for _idx in range(len(target_end_pos)):
                    ans_tokens.append([vocab.itos[int(seq_input[num][i])]
                                       for i in range(target_start_pos[_idx][num],
                                                      target_end_pos[_idx][num] + 1)])
                pred_tokens = [vocab.itos[int(seq_input[num][i])]
                               for i in range(start_pos[num],
                                              end_pos[num] + 1)]
                ans_pred_tokens_samples.append((ans_tokens, pred_tokens))
    return total_loss / (len(data_source) // batch_size), \
        compute_qa_exact(ans_pred_tokens_samples), \
        compute_qa_f1(ans_pred_tokens_samples)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_batch)
    train_loss_log.append(0.0)
    for idx, (seq_input, ans_pos, tok_type) in enumerate(dataloader):
        optimizer.zero_grad()
        start_pos, end_pos = model(seq_input, token_type_input=tok_type)
        target_start_pos, target_end_pos = ans_pos[0].to(device).split(1, dim=-1)
        target_start_pos = target_start_pos.squeeze(-1)
        target_end_pos = target_end_pos.squeeze(-1)
        loss = (criterion(start_pos, target_start_pos) + criterion(end_pos, target_end_pos)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            train_loss_log[-1] = cur_loss
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, idx,
                                                      len(train_dataset) // batch_size,
                                                      scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / args.log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question-Answer fine-tuning task')
    parser.add_argument('--lr', type=float, default=5.0,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=72, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='max. sequence length for context + question')
    parser.add_argument('--seed', type=int, default=21192391,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='qa_model.pt',
                        help='path to save the final bert model')
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str, default='ns_bert.pt',
                        help='path to save the pretrained bert')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, dev_dataset = SQuAD1()
        old_vocab = train_dataset.vocab
        vocab = torchtext.legacy.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)
    pad_id = vocab.stoi['<pad>']
    sep_id = vocab.stoi['<sep>']
    cls_id = vocab.stoi['<cls>']
    train_dataset, dev_dataset = SQuAD1(vocab=vocab)
    train_dataset = process_raw_data(train_dataset)
    dev_dataset = process_raw_data(dev_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_layer = BertEmbedding(len(vocab), args.emsize)
    pretrained_bert = BertModel(len(vocab), args.emsize, args.nhead, args.nhid, args.nlayers, embed_layer, args.dropout)
    pretrained_bert.load_state_dict(torch.load(args.bert_model))
    model = QuestionAnswerTask(pretrained_bert).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    best_f1 = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, val_exact, val_f1 = evaluate(dev_dataset, vocab)
        val_loss_log.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'exact {:8.3f}% | '
              'f1 {:8.3f}%'.format(epoch, (time.time() - epoch_start_time),
                                   val_loss, val_exact, val_f1))
        print('-' * 89)
        if best_f1 is None or val_f1 > best_f1:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_f1 = val_f1
        else:
            scheduler.step()

    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_loss, test_exact, test_f1 = evaluate(dev_dataset, vocab)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | exact {:8.3f}% | f1 {:8.3f}%'.format(
        test_loss, test_exact, test_f1))
    print('=' * 89)
    print_loss_log('qa_loss.txt', train_loss_log, val_loss_log, test_loss)
    with open(args.save, 'wb') as f:
        torch.save(model, f)


#!/usr/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import unittest

from torchtext.legacy import data
from torchtext.legacy.datasets import TREC


class TestSubword(unittest.TestCase):
    def test_subword_trec(self):
        TEXT = data.SubwordField()
        LABEL = data.Field(sequential=False)
        RAW = data.Field(sequential=False, use_vocab=False)
        raw, _ = TREC.splits(RAW, LABEL)
        cooked, _ = TREC.splits(TEXT, LABEL)
        LABEL.build_vocab(cooked)
        TEXT.build_vocab(cooked, max_size=100)
        TEXT.segment(cooked)
        print(cooked[0].text)
        batch = next(iter(data.Iterator(cooked, 1, shuffle=False)))
        self.assertEqual(TEXT.reverse(batch.text.data)[0], raw[0].text)


if __name__ == '__main__':
    unittest.main()


import unittest

import torch
import torchaudio.transforms as T
from torchaudio._internal.module_utils import is_module_available
from parameterized import param, parameterized

from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    get_whitenoise,
    get_sinusoid,
    get_spectrogram,
    nested_params,
)

LIBROSA_AVAILABLE = is_module_available('librosa')

if LIBROSA_AVAILABLE:
    import librosa


@unittest.skipIf(not LIBROSA_AVAILABLE, "Librosa not available")
class TransformsTestBase(TestBaseMixin):
    @parameterized.expand([
        param(n_fft=400, hop_length=200, power=2.0),
        param(n_fft=600, hop_length=100, power=2.0),
        param(n_fft=400, hop_length=200, power=3.0),
        param(n_fft=200, hop_length=50, power=2.0),
    ])
    def test_Spectrogram(self, n_fft, hop_length, power):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.core.spectrum._spectrogram(
            y=waveform[0].cpu().numpy(),
            n_fft=n_fft, hop_length=hop_length, power=power)[0]

        result = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=power,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    def test_Spectrogram_complex(self):
        n_fft = 400
        hop_length = 200
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.core.spectrum._spectrogram(
            y=waveform[0].cpu().numpy(),
            n_fft=n_fft, hop_length=hop_length, power=1)[0]

        result = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, return_complex=True,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result.abs(), torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    @nested_params(
        [
            param(n_fft=400, hop_length=200, n_mels=64),
            param(n_fft=600, hop_length=100, n_mels=128),
            param(n_fft=200, hop_length=50, n_mels=32),
        ],
        [param(norm=norm) for norm in [None, 'slaney']],
        [param(mel_scale=mel_scale) for mel_scale in ['htk', 'slaney']],
    )
    def test_MelSpectrogram(self, n_fft, hop_length, n_mels, norm, mel_scale):
        sample_rate = 16000
        waveform = get_sinusoid(
            sample_rate=sample_rate, n_channels=1,
        ).to(self.device, self.dtype)

        expected = librosa.feature.melspectrogram(
            y=waveform[0].cpu().numpy(),
            sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels, norm=norm,
            htk=mel_scale == "htk")
        result = T.MelSpectrogram(
            sample_rate=sample_rate, window_fn=torch.hann_window,
            hop_length=hop_length, n_mels=n_mels,
            n_fft=n_fft, norm=norm, mel_scale=mel_scale,
        ).to(self.device, self.dtype)(waveform)[0]
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)

    def test_magnitude_to_db(self):
        spectrogram = get_spectrogram(
            get_whitenoise(), n_fft=400, power=2).to(self.device, self.dtype)
        result = T.AmplitudeToDB('magnitude', 80.).to(self.device, self.dtype)(spectrogram)[0]
        expected = librosa.core.spectrum.amplitude_to_db(spectrogram[0].cpu().numpy())
        self.assertEqual(result, torch.from_numpy(expected))

    def test_power_to_db(self):
        spectrogram = get_spectrogram(
            get_whitenoise(), n_fft=400, power=2).to(self.device, self.dtype)
        result = T.AmplitudeToDB('power', 80.).to(self.device, self.dtype)(spectrogram)[0]
        expected = librosa.core.spectrum.power_to_db(spectrogram[0].cpu().numpy())
        self.assertEqual(result, torch.from_numpy(expected))

    @nested_params([
        param(n_fft=400, hop_length=200, n_mels=64, n_mfcc=40),
        param(n_fft=600, hop_length=100, n_mels=128, n_mfcc=20),
        param(n_fft=200, hop_length=50, n_mels=32, n_mfcc=25),
    ])
    def test_mfcc(self, n_fft, hop_length, n_mels, n_mfcc):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1).to(self.device, self.dtype)
        result = T.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc, norm='ortho',
            melkwargs={'hop_length': hop_length, 'n_fft': n_fft, 'n_mels': n_mels},
        ).to(self.device, self.dtype)(waveform)[0]

        melspec = librosa.feature.melspectrogram(
            y=waveform[0].cpu().numpy(), sr=sample_rate, n_fft=n_fft,
            win_length=n_fft, hop_length=hop_length,
            n_mels=n_mels, htk=True, norm=None)
        expected = librosa.feature.mfcc(
            S=librosa.core.spectrum.power_to_db(melspec),
            n_mfcc=n_mfcc, dct_type=2, norm='ortho')
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)

    @parameterized.expand([
        param(n_fft=400, hop_length=200),
        param(n_fft=600, hop_length=100),
        param(n_fft=200, hop_length=50),
    ])
    def test_spectral_centroid(self, n_fft, hop_length):
        sample_rate = 16000
        waveform = get_whitenoise(
            sample_rate=sample_rate, n_channels=1).to(self.device, self.dtype)

        result = T.SpectralCentroid(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        ).to(self.device, self.dtype)(waveform)
        expected = librosa.feature.spectral_centroid(
            y=waveform[0].cpu().numpy(), sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.assertEqual(result, torch.from_numpy(expected), atol=5e-4, rtol=1e-5)


from pathlib import Path
from typing import Union, Tuple, List

import torch
from torch.utils.data import Dataset

import torchaudio

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]


class LibriMix(Dataset):
    r"""Create the LibriMix dataset.

    Args:
        root (str or Path): The path to the directory where the directory ``Libri2Mix`` or
            ``Libri3Mix`` is stored.
        subset (str, optional): The subset to use. Options: [``train-360`, ``train-100``,
            ``dev``, and ``test``] (Default: ``train-360``).
        num_speakers (int, optional): The number of speakers, which determines the directories
            to traverse. The Dataset will traverse ``s1`` to ``sN`` directories to collect
            N source audios. (Default: 2)
        sample_rate (int, optional): sample rate of audio files. The ``sample_rate`` determines
            which subdirectory the audio are fetched. If any of the audio has a different sample
            rate, raises ``ValueError``. Options: [8000, 16000] (Default: 8000)
        task (str, optional): the task of LibriMix.
            Options: [``enh_single``, ``enh_both``, ``sep_clean``, ``sep_noisy``]
            (Default: ``sep_clean``)

    Note:
        The LibriMix dataset needs to be manually generated. Please check https://github.com/JorisCos/LibriMix
    """
    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        task: str = "sep_clean",
    ):
        self.root = Path(root) / f"Libri{num_speakers}Mix"
        if sample_rate == 8000:
            self.root = self.root / "wav8k/min" / subset
        elif sample_rate == 16000:
            self.root = self.root / "wav16k/min" / subset
        else:
            raise ValueError(
                f"Unsupported sample rate. Found {sample_rate}."
            )
        self.sample_rate = sample_rate
        self.task = task
        self.mix_dir = (self.root / f"mix_{task.split('_')[1]}").resolve()
        self.src_dirs = [(self.root / f"s{i+1}").resolve() for i in range(num_speakers)]

        self.files = [p.name for p in self.mix_dir.glob("*wav")]
        self.files.sort()

    def _load_audio(self, path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"The dataset contains audio file of sample rate {sample_rate}, "
                f"but the requested sample rate is {self.sample_rate}."
            )
        return waveform

    def _load_sample(self, filename) -> SampleType:
        mixed = self._load_audio(str(self.mix_dir / filename))
        srcs = []
        for i, dir_ in enumerate(self.src_dirs):
            src = self._load_audio(str(dir_ / filename))
            if mixed.shape != src.shape:
                raise ValueError(
                    f"Different waveform shapes. mixed: {mixed.shape}, src[{i}]: {src.shape}"
                )
            srcs.append(src)
        return self.sample_rate, mixed, srcs

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return self._load_sample(self.files[key])


import torch

from .tacotron2_loss_impl import (
    Tacotron2LossShapeTests,
    Tacotron2LossTorchscriptTests,
    Tacotron2LossGradcheckTests,
)
from torchaudio_unittest.common_utils import skipIfNoCuda, PytorchTestCase


@skipIfNoCuda
class TestTacotron2LossShapeFloat32CUDA(PytorchTestCase, Tacotron2LossShapeTests):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2TorchsciptFloat32CUDA(PytorchTestCase, Tacotron2LossTorchscriptTests):
    dtype = torch.float32
    device = torch.device("cuda")


@skipIfNoCuda
class TestTacotron2GradcheckFloat64CUDA(PytorchTestCase, Tacotron2LossGradcheckTests):
    dtype = torch.float64   # gradcheck needs a higher numerical accuracy
    device = torch.device("cuda")




from .common_utils import create_tsv
from .feature_utils import dump_features
from .kmeans import learn_kmeans, get_km_label

__all__ = [
    "create_tsv",
    "dump_features",
    "learn_kmeans",
    "get_km_label",
]


import os
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _find_match,
    _create_dataset_directory,
    _create_data_from_csv,
)

URL = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg'

MD5 = '620c8ae4bd5a150b730f1ba9a7c6a4d3'

NUM_LINES = {
    'train': 560000,
    'test': 38000,
}

_PATH = 'yelp_review_polarity_csv.tar.gz'

DATASET_NAME = "YelpReviewPolarity"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'test'))
def YelpReviewPolarity(root, split):
    dataset_tar = download_from_url(URL, root=root,
                                    path=os.path.join(root, _PATH),
                                    hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    path = _find_match(split + '.csv', extracted_files)
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split],
                                   _create_data_from_csv(path))


import argparse
from collections import (Counter, OrderedDict)
import time
import random
import string
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import torch
from torchtext.datasets import DATASETS
from torchtext.experimental.vocab_factory import (
    load_vocab_from_file,
    build_vocab_from_text_file
)
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab as VocabNew
from torchtext.legacy.vocab import (
    Vocab,
    build_vocab_from_iterator as build_vocab_from_iterator_legacy,
)
from torchtext.experimental.transforms import(
    basic_english_normalize,
)
from torchtext.data.utils import get_tokenizer

def build_vocab(data, transforms):
    def apply_transforms(data):
        for _, line in data:
            yield transforms(line)
    vocab = build_vocab_from_iterator(apply_transforms(data), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def compare_legacy_and_new_batch_lookup():
    num_tokens = 1000
    num_letters = 6
    num_lines = 100000
    vocab = [''.join(random.sample(string.ascii_letters * num_letters, num_letters)) for _ in range(num_tokens)]
    counter = Counter()
    counter.update(vocab)
    legacy_vocab = Vocab(counter)
    new_vocab = VocabNew(counter)
    speed_ups = []
    token_lengths = [i for i in range(2, 100)]
    for i in token_lengths:
        lines = [random.sample(vocab, i) for _ in range(num_lines)]
        start_time = timer()
        for text in lines:
            legacy_vocab.lookup_indices(text)
        legacy_time = timer() - start_time

        start_time = timer()
        for text in lines:
            new_vocab.lookup_indices(text)

        new_time = timer() - start_time

        speed_ups.append(legacy_time / new_time)
        print("speed-up={} for average length={}".format(legacy_time / new_time, i))
        del lines

    plt.close()
    fig, ax = plt.subplots(1, 1)
    ax.plot(token_lengths, speed_ups)
    ax.set_xlabel('Average Tokens per line')
    ax.set_ylabel('Speed-up')
    plt.savefig("speedup.jpg")


def legacy_vocab_from_file_object(file_like_object, **kwargs):
    r"""Create a `Vocab` object from a file like object.

    The `file_like_object` should contain tokens seperated by new lines. Note that the vocab
    will be created in the order that the tokens first appear in the file (and not by the frequency of tokens).

    Format for txt file:
        token1
        token2
        ...
        token_n

    Args:
        file_like_object (FileObject): a file like object to read data from.
        Remaining keyword arguments: Passed to the constructor of Vocab class.

    Returns:
        Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> v = vocab_from_file_object(f, specials=('<unk>', '<pad>', '<eos>'), specials_first=False)
    """
    tokenizer = basic_english_normalize()

    def tokenize(line):
        return tokenizer(line)

    def token_iterator(lines):
        for line in lines:
            for token in tokenize(line):
                yield token

    return build_vocab_from_iterator_legacy(token_iterator(file_like_object))


def benchmark_new_vocab_construction(vocab_file_path, is_raw_text=True, is_legacy=True, num_iters=1):
    f = open(vocab_file_path, 'r')
    t0 = time.monotonic()
    if is_raw_text:
        if is_legacy:
            print("Loading from raw text file with legacy python function")
            for _ in range(num_iters):
                legacy_vocab_from_file_object(f)

            print("Construction time:", time.monotonic() - t0)
        else:
            print("Loading from raw text file with basic_english_normalize tokenizer")
            for _ in range(num_iters):
                tokenizer = basic_english_normalize()
                jited_tokenizer = torch.jit.script(tokenizer)
                build_vocab_from_text_file(vocab_file_path, jited_tokenizer, num_cpus=1)
            print("Construction time:", time.monotonic() - t0)
    else:
        for _ in range(num_iters):
            load_vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)


def benchmark_new_vocab_lookup(vocab_file_path=None, dataset='AG_NEWS'):
    def _run_benchmark_lookup(tokens, vocab):
        t0 = time.monotonic()
        # list lookup
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            for tokens_list in tokens:
                vocab.lookup_indices(tokens_list)
        # single token lookup
        elif isinstance(tokens, list):
            for token in tokens:
                vocab[token]
        else:
            raise RuntimeError("Received tokens of incorrect type {}.".format(type(tokens)))
        print("Lookup time:", time.monotonic() - t0)

    tokens = []
    tokens_lists = []
    tokenizer = get_tokenizer("basic_english")
    for (_, text) in DATASETS[dataset](split='train'):
       cur_tokens = tokenizer(text)
       tokens_lists.append(cur_tokens)
       tokens += cur_tokens

    if vocab_file_path:
        print("Loading Vocab from file {}".format(vocab_file_path))

        def token_iterator(file_path):
            f = open(file_path, 'r')
            for token in f:
                yield token

        # existing Vocab construction
        print("Vocab")
        t0 = time.monotonic()
        v_existing = build_vocab_from_iterator_legacy(token_iterator(vocab_file_path))
        print("Construction time:", time.monotonic() - t0)

        # new Vocab construction
        print("Vocab New")
        t0 = time.monotonic()
        f = open(vocab_file_path, 'r')
        v_new = load_vocab_from_file(f)
        print("Construction time:", time.monotonic() - t0)
    else:
        print("Loading Vocab from {}".format(dataset))
        counter = Counter(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        # existing Vocab construction
        print("Vocab")
        t0 = time.monotonic()
        v_existing = Vocab(counter)
        print("Construction time:", time.monotonic() - t0)

        # new Vocab construction
        print("Vocab New")
        t0 = time.monotonic()
        v_new = VocabNew(ordered_dict)
        print("Construction time:", time.monotonic() - t0)
    jit_v_new = torch.jit.script(v_new)

    # existing Vocab eager lookup
    print("Vocab - Eager Mode")
    _run_benchmark_lookup(tokens, v_existing)
    _run_benchmark_lookup([tokens], v_existing)
    _run_benchmark_lookup(tokens_lists, v_existing)

    # new Vocab eager lookup
    print("Vocab New - Eager Mode")
    _run_benchmark_lookup(tokens, v_new)
    _run_benchmark_lookup([tokens], v_new)
    _run_benchmark_lookup(tokens_lists, v_new)

    jit_v_new = torch.jit.script(v_new)
    # new Vocab jit lookup
    print("Vocab New - Jit Mode")
    _run_benchmark_lookup(tokens, jit_v_new)
    _run_benchmark_lookup([tokens], jit_v_new)
    _run_benchmark_lookup(tokens_lists, jit_v_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data procesing pipelines')
    parser.add_argument('--run-construction-benchmark', type=bool, default=False,
                        help='run benchmark for constructing a vocab (default=False)')
    parser.add_argument('--is-raw-text', type=bool, default=True,
                        help='construct vocab from raw text file (default=True)')
    parser.add_argument('--is-legacy', type=bool, default=False,
                        help='construct vocab using legacy implementation (default=False)')
    parser.add_argument('--vocab-filename-construction', type=str, default='vocab.txt',
                        help='The name of vocab file used for construction')
    parser.add_argument('--vocab-filename-lookup', type=str, default=None,
                        help='The name of vocab file used for lookup')
    parser.add_argument('--dataset', type=str, default='AG_NEWS',
                        help='The name of vocab file used for lookup')
    args = parser.parse_args()

    if args.run_construction_benchmark:
        print("is_legacy", args.is_legacy)
        benchmark_new_vocab_construction(args.vocab_filename_construction,
                                                  is_raw_text=args.is_raw_text, is_legacy=args.is_legacy)
    else:
        benchmark_new_vocab_lookup(args.vocab_filename_lookup, args.dataset)


import math

import torch
from torch import nn as nn
from torch.nn import functional as F


class LongCrossEntropyLoss(nn.Module):
    r""" CrossEntropy loss
    """

    def __init__(self):
        super(LongCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        output = output.transpose(1, 2)
        target = target.long()

        criterion = nn.CrossEntropyLoss()
        return criterion(output, target)


class MoLLoss(nn.Module):
    r""" Discretized mixture of logistic distributions loss

    Adapted from wavenet vocoder
    (https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py)
    Explanation of loss (https://github.com/Rayhane-mamah/Tacotron-2/issues/155)

    Args:
        y_hat (Tensor): Predicted output (n_batch x n_time x n_channel)
        y (Tensor): Target (n_batch x n_time x 1)
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each minibatch

    Returns
        Tensor: loss
    """

    def __init__(self, num_classes=65536, log_scale_min=None, reduce=True):
        super(MoLLoss, self).__init__()
        self.num_classes = num_classes
        self.log_scale_min = log_scale_min
        self.reduce = reduce

    def forward(self, y_hat, y):
        y = y.unsqueeze(-1)

        if self.log_scale_min is None:
            self.log_scale_min = math.log(1e-14)

        assert y_hat.dim() == 3
        assert y_hat.size(-1) % 3 == 0

        nr_mix = y_hat.size(-1) // 3

        # unpack parameters (n_batch, n_time, num_mixtures) x 3
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix: 2 * nr_mix]
        log_scales = torch.clamp(
            y_hat[:, :, 2 * nr_mix: 3 * nr_mix], min=self.log_scale_min
        )

        # (n_batch x n_time x 1) to (n_batch x n_time x num_mixtures)
        y = y.expand_as(means)

        centered_y = y - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1.0 / (self.num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1.0 / (self.num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0 (before scaling)
        # equivalent: torch.log(F.sigmoid(plus_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)

        # log probability for edge case of 255 (before scaling)
        # equivalent: (1 - F.sigmoid(min_in)).log()
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered_y
        # log probability in the center of the bin, to be used in extreme cases
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

        inner_inner_cond = (cdf_delta > 1e-5).float()

        inner_inner_out = inner_inner_cond * torch.log(
            torch.clamp(cdf_delta, min=1e-12)
        ) + (1.0 - inner_inner_cond) * (
            log_pdf_mid - math.log((self.num_classes - 1) / 2)
        )
        inner_cond = (y > 0.999).float()
        inner_out = (
            inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
        )
        cond = (y < -0.999).float()
        log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out

        log_probs = log_probs + F.log_softmax(logit_probs, -1)

        if self.reduce:
            return -torch.mean(_log_sum_exp(log_probs))
        else:
            return -_log_sum_exp(log_probs).unsqueeze(-1)


def _log_sum_exp(x):
    r""" Numerically stable log_sum_exp implementation that prevents overflow
    """

    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


from parameterized import parameterized
import torch
import torchaudio.functional as F

from torchaudio_unittest.common_utils import (
    get_sinusoid,
    load_params,
    save_wav,
    skipIfNoExec,
    TempDirMixin,
    TestBaseMixin,
)
from torchaudio_unittest.common_utils.kaldi_utils import (
    convert_args,
    run_kaldi,
)


class Kaldi(TempDirMixin, TestBaseMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)

    @skipIfNoExec('apply-cmvn-sliding')
    def test_sliding_window_cmn(self):
        """sliding_window_cmn should be numerically compatible with apply-cmvn-sliding"""
        kwargs = {
            'cmn_window': 600,
            'min_cmn_window': 100,
            'center': False,
            'norm_vars': False,
        }

        tensor = torch.randn(40, 10, dtype=self.dtype, device=self.device)
        result = F.sliding_window_cmn(tensor, **kwargs)
        command = ['apply-cmvn-sliding'] + convert_args(**kwargs) + ['ark:-', 'ark:-']
        kaldi_result = run_kaldi(command, 'ark', tensor)
        self.assert_equal(result, expected=kaldi_result)


class KaldiCPUOnly(TempDirMixin, TestBaseMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)

    @parameterized.expand(load_params('kaldi_test_pitch_args.jsonl'))
    @skipIfNoExec('compute-kaldi-pitch-feats')
    def test_pitch_feats(self, kwargs):
        """compute_kaldi_pitch produces numerically compatible result with compute-kaldi-pitch-feats"""
        sample_rate = kwargs['sample_rate']
        waveform = get_sinusoid(dtype='float32', sample_rate=sample_rate)
        result = F.compute_kaldi_pitch(waveform[0], **kwargs)

        waveform = get_sinusoid(dtype='int16', sample_rate=sample_rate)
        wave_file = self.get_temp_path('test.wav')
        save_wav(wave_file, waveform, sample_rate)

        command = ['compute-kaldi-pitch-feats'] + convert_args(**kwargs) + ['scp:-', 'ark:-']
        kaldi_result = run_kaldi(command, 'scp', wave_file)
        self.assert_equal(result, expected=kaldi_result)


import os
from pathlib import Path
from typing import List, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "waves_yesno",
        "url": "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
        "checksum": "c3f49e0cca421f96b75b41640749167b52118f232498667ca7a5f9416aef8e73",
    }
}


class YESNO(Dataset):
    """Create a Dataset for YesNo.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"http://www.openslr.org/resources/1/waves_yesno.tar.gz"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"waves_yesno"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = _RELEASE_CONFIGS["release1"]["url"],
        folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        download: bool = False
    ) -> None:

        self._parse_filesystem(root, url, folder_in_archive, download)

    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        root = Path(root)
        archive = os.path.basename(url)
        archive = root / archive

        self._path = root / folder_in_archive
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS["release1"]["checksum"]
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*.wav"))

    def _load_item(self, fileid: str, path: str):
        labels = [int(c) for c in fileid.split("_")]
        file_audio = os.path.join(path, fileid + ".wav")
        waveform, sample_rate = torchaudio.load(file_audio)
        return waveform, sample_rate, labels

    def __getitem__(self, n: int) -> Tuple[Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, List[int]): ``(waveform, sample_rate, labels)``
        """
        fileid = self._walker[n]
        item = self._load_item(fileid, self._path)
        return item

    def __len__(self) -> int:
        return len(self._walker)


import torch


class Decoder(torch.nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def forward(self, logits: torch.Tensor) -> str:
        """Given a sequence logits over labels, get the best path string

        Args:
            logits (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        best_path = torch.argmax(logits, dim=-1)  # [num_seq,]
        best_path = torch.unique_consecutive(best_path, dim=-1)
        hypothesis = ''
        for i in best_path:
            char = self.labels[i]
            if char in ['<s>', '<pad>']:
                continue
            if char == '|':
                char = ' '
            hypothesis += char
        return hypothesis


from pathlib import Path
from typing import Dict, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    validate_file,
)


_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"
_CHECKSUM = "29e93debeb0e779986542229a81ff29b"
_SUPPORTED_SUBSETS = {"train", "test"}


class DR_VCTK(Dataset):
    """Create a dataset for Device Recorded VCTK (Small subset version).

    Args:
        root (str or Path): Root directory where the dataset's top level directory is found.
        subset (str): The subset to use. Can be one of ``"train"`` and ``"test"``. (default: ``"train"``).
        download (bool):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str): The URL to download the dataset from.
            (default: ``"https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train",
        *,
        download: bool = False,
        url: str = _URL,
    ) -> None:
        if subset not in _SUPPORTED_SUBSETS:
            raise RuntimeError(
                f"The subset '{subset}' does not match any of the supported subsets: {_SUPPORTED_SUBSETS}"
            )

        root = Path(root).expanduser()
        archive = root / "DR-VCTK.zip"

        self._subset = subset
        self._path = root / "DR-VCTK" / "DR-VCTK"
        self._clean_audio_dir = self._path / f"clean_{self._subset}set_wav_16k"
        self._noisy_audio_dir = self._path / f"device-recorded_{self._subset}set_wav_16k"
        self._config_filepath = self._path / "configurations" / f"{self._subset}_ch_log.txt"

        if not self._path.is_dir():
            if not archive.is_file():
                if not download:
                    raise RuntimeError("Dataset not found. Please use `download=True` to download it.")
                download_url(url, root)
            self._validate_checksum(archive)
            extract_archive(archive, root)

        self._config = self._load_config(self._config_filepath)
        self._filename_list = sorted(self._config)

    def _validate_checksum(self, archive):
        with open(archive, "rb") as file_obj:
            if not validate_file(file_obj, _CHECKSUM, "md5"):
                raise RuntimeError(
                    f"The hash of {str(archive)} does not match. Delete the file manually and retry."
                )

    def _load_config(self, filepath: str) -> Dict[str, Tuple[str, int]]:
        # Skip header
        skip_rows = 2 if self._subset == "train" else 1

        config = {}
        with open(filepath) as f:
            for i, line in enumerate(f):
                if i < skip_rows or not line:
                    continue
                filename, source, channel_id = line.strip().split("\t")
                config[filename] = (source, int(channel_id))
        return config

    def _load_dr_vctk_item(self, filename: str) -> Tuple[Tensor, int, Tensor, int, str, str, str, int]:
        speaker_id, utterance_id = filename.split(".")[0].split("_")
        source, channel_id = self._config[filename]
        file_clean_audio = self._clean_audio_dir / filename
        file_noisy_audio = self._noisy_audio_dir / filename
        waveform_clean, sample_rate_clean = torchaudio.load(file_clean_audio)
        waveform_noisy, sample_rate_noisy = torchaudio.load(file_noisy_audio)
        return (
            waveform_clean,
            sample_rate_clean,
            waveform_noisy,
            sample_rate_noisy,
            speaker_id,
            utterance_id,
            source,
            channel_id,
        )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, Tensor, int, str, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, Tensor, int, str, str, str, int):
            ``(waveform_clean, sample_rate_clean, waveform_noisy, sample_rate_noisy, speaker_id,\
                utterance_id, source, channel_id)``
        """
        filename = self._filename_list[n]
        return self._load_dr_vctk_item(filename)

    def __len__(self) -> int:
        return len(self._filename_list)


import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .kaldi_compatibility_test_impl import Kaldi, KaldiCPUOnly


class TestKaldiCPUOnly(KaldiCPUOnly, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestKaldiFloat32(Kaldi, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestKaldiFloat64(Kaldi, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


import torch


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Deprecated: this attribute is left for backwards compatibility,
            however it is UNUSED as of the merger with pytorch 0.4.
        input_fields: The names of the fields that are used as input for the model
        target_fields: The names of the fields that are used as targets during
                       model training

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=None, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.fields = dataset.fields.keys()
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.__dict__:
            return 'Empty {} instance'.format(torch.typename(self))

        fields_to_index = filter(lambda field: field is not None, self.fields)
        var_strs = '\n'.join(['\t[.' + name + ']' + ":" + _short_str(getattr(self, name))
                              for name in fields_to_index if hasattr(self, name)])

        data_str = (' from {}'.format(self.dataset.name.upper())
                    if hasattr(self.dataset, 'name')
                    and isinstance(self.dataset.name, str) else '')

        strt = '[{} of size {}{}]\n{}'.format(torch.typename(self),
                                              self.batch_size, data_str, var_strs)
        return '\n' + strt

    def __len__(self):
        return self.batch_size

    def _get_field_values(self, fields):
        if len(fields) == 0:
            return None
        elif len(fields) == 1:
            return getattr(self, fields[0])
        else:
            return tuple(getattr(self, f) for f in fields)

    def __iter__(self):
        yield self._get_field_values(self.input_fields)
        yield self._get_field_values(self.target_fields)


def _short_str(tensor):
    # unwrap variable to tensor
    if not torch.is_tensor(tensor):
        # (1) unpack variable
        if hasattr(tensor, 'data'):
            tensor = tensor.data
        # (2) handle include_lengths
        elif isinstance(tensor, tuple):
            return str(tuple(_short_str(t) for t in tensor))
        # (3) fallback to default str
        else:
            return str(tensor)

    # copied from torch _tensor_str
    size_str = 'x'.join(str(size) for size in tensor.size())
    device_str = '' if not tensor.is_cuda else \
        ' (GPU {})'.format(tensor.get_device())
    strt = '[{} of size {}{}]'.format(torch.typename(tensor),
                                      size_str, device_str)
    return strt


# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Tuple

from torch import nn, Tensor


class Tacotron2Loss(nn.Module):
    """Tacotron2 loss function modified from:
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/loss_function.py
    """

    def __init__(self):
        super().__init__()

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        model_outputs: Tuple[Tensor, Tensor, Tensor],
        targets: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Pass the input through the Tacotron2 loss.

        The original implementation was introduced in
        *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*
        [:footcite:`shen2018natural`].

        Args:
            model_outputs (tuple of three Tensors): The outputs of the
                Tacotron2. These outputs should include three items:
                (1) the predicted mel spectrogram before the postnet (``mel_specgram``)
                    with shape (batch, mel, time).
                (2) predicted mel spectrogram after the postnet (``mel_specgram_postnet``)
                    with shape (batch, mel, time), and
                (3) the stop token prediction (``gate_out``) with shape (batch, ).
            targets (tuple of two Tensors): The ground truth mel spectrogram (batch, mel, time) and
                stop token with shape (batch, ).

        Returns:
            mel_loss (Tensor): The mean MSE of the mel_specgram and ground truth mel spectrogram
                with shape ``torch.Size([])``.
            mel_postnet_loss (Tensor): The mean MSE of the mel_specgram_postnet and
                ground truth mel spectrogram with shape ``torch.Size([])``.
            gate_loss (Tensor): The mean binary cross entropy loss of
                the prediction on the stop token with shape ``torch.Size([])``.
        """
        mel_target, gate_target = targets[0], targets[1]
        gate_target = gate_target.view(-1, 1)

        mel_specgram, mel_specgram_postnet, gate_out = model_outputs
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_specgram, mel_target)
        mel_postnet_loss = self.mse_loss(mel_specgram_postnet, mel_target)
        gate_loss = self.bce_loss(gate_out, gate_target)
        return mel_loss, mel_postnet_loss, gate_loss


import torch

from torchaudio_unittest.common_utils import PytorchTestCase
from .torchscript_consistency_impl import Transforms, TransformsFloat32Only, TransformsFloat64Only


class TestTransformsFloat32(Transforms, TransformsFloat32Only, PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestTransformsFloat64(Transforms, TransformsFloat64Only, PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


"""
Speech Recognition with Wav2Vec2
================================

**Author**: `Moto Hira <moto@fb.com>`__

This tutorial shows how to perform speech recognition using using
pre-trained models from wav2vec 2.0
[`paper <https://arxiv.org/abs/2006.11477>`__].

"""


######################################################################
# Overview
# --------
#
# The process of speech recognition looks like the following.
#
# 1. Extract the acoustic features from audio waveform
#
# 2. Estimate the class of the acoustic features frame-by-frame
#
# 3. Generate hypothesis from the sequence of the class probabilities
#
# Torchaudio provides easy access to the pre-trained weights and
# associated information, such as the expected sample rate and class
# labels. They are bundled together and available under
# ``torchaudio.pipelines`` module.
#


######################################################################
# Preparation
# -----------
#
# First we import the necessary packages, and fetch data that we work on.
#

# %matplotlib inline

import os

import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
SPEECH_FILE = "_assets/speech.wav"

if not os.path.exists(SPEECH_FILE):
    os.makedirs("_assets", exist_ok=True)
    with open(SPEECH_FILE, "wb") as file:
        file.write(requests.get(SPEECH_URL).content)


######################################################################
# Creating a pipeline
# -------------------
#
# First, we will create a Wav2Vec2 model that performs the feature
# extraction and the classification.
#
# There are two types of Wav2Vec2 pre-trained weights available in
# torchaudio. The ones fine-tuned for ASR task, and the ones not
# fine-tuned.
#
# Wav2Vec2 (and HuBERT) models are trained in self-supervised manner. They
# are firstly trained with audio only for representation learning, then
# fine-tuned for a specific task with additional labels.
#
# The pre-trained weights without fine-tuning can be fine-tuned
# for other downstream tasks as well, but this tutorial does not
# cover that.
#
# We will use :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` here.
#
# There are multiple models available as
# :py:mod:`torchaudio.pipelines`. Please check the documentation for
# the detail of how they are trained.
#
# The bundle object provides the interface to instantiate model and other
# information. Sampling rate and the class labels are found as follow.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())


######################################################################
# Model can be constructed as following. This process will automatically
# fetch the pre-trained weights and load it into the model.
#

model = bundle.get_model().to(device)

print(model.__class__)


######################################################################
# Loading data
# ------------
#
# We will use the speech data from `VOiCES
# dataset <https://iqtlabs.github.io/voices/>`__, which is licensed under
# Creative Commos BY 4.0.
#

IPython.display.Audio(SPEECH_FILE)


######################################################################
# To load data, we use :py:func:`torchaudio.load`.
#
# If the sampling rate is different from what the pipeline expects, then
# we can use :py:func:`torchaudio.functional.resample` for resampling.
#
# .. note::
#
#    - :py:func:`torchaudio.functional.resample` works on CUDA tensors as well.
#    - When performing resampling multiple times on the same set of sample rates,
#      using :py:func:`torchaudio.transforms.Resample` might improve the performace.
#

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


######################################################################
# Extracting acoustic features
# ----------------------------
#
# The next step is to extract acoustic features from the audio.
#
# .. note::
#    Wav2Vec2 models fine-tuned for ASR task can perform feature
#    extraction and classification with one step, but for the sake of the
#    tutorial, we also show how to perform feature extraction here.
#

with torch.inference_mode():
    features, _ = model.extract_features(waveform)


######################################################################
# The returned features is a list of tensors. Each tensor is the output of
# a transformer layer.
#

fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu())
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
plt.tight_layout()
plt.show()


######################################################################
# Feature classification
# ----------------------
#
# Once the acoustic features are extracted, the next step is to classify
# them into a set of categories.
#
# Wav2Vec2 model provides method to perform the feature extraction and
# classification in one step.
#

with torch.inference_mode():
    emission, _ = model(waveform)


######################################################################
# The output is in the form of logits. It is not in the form of
# probability.
#
# Lets visualize this.
#

plt.imshow(emission[0].cpu().T)
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.show()
print("Class labels:", bundle.get_labels())


######################################################################
# We can see that there are strong indications to certain labels across
# the time line.
#


######################################################################
# Generating transcripts
# ----------------------
#
# From the sequence of label probabilities, now we want to generate
# transcripts. The process to generate hypotheses is often called
# decoding.
#
# Decoding is more elaborate than simple classification because
# decoding at certain time step can be affected by surrounding
# observations.
#
# For example, take a word like ``night`` and ``knight``. Even if their
# prior probability distribution are differnt (in typical conversations,
# ``night`` would occur way more often than ``knight``), to accurately
# generate transcripts with ``knight``, such as ``a knight with a sword``,
# the decoding process has to postpone the final decision until it sees
# enough context.
#
# There are many decoding techniques proposed, and they require external
# resources, such as word dictionary and language models.
#
# In this tutorial, for the sake of simplicity, we will perform greedy
# decoding which does not depend on such external components, and simply
# pick up the best hypothesis at each time step. Therefore, the context
# information are not used, and only one transcript can be generated.
#
# We start by defining greedy decoding algorithm.
#


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


######################################################################
# Now create the decoder object and decode the transcript.
#

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])


######################################################################
# Lets check the result and listen again to the audio.
#

print(transcript)
IPython.display.Audio(SPEECH_FILE)


######################################################################
# The ASR model is fine-tuned using a loss function called Connectionist Temporal Classification (CTC).
# The detail of CTC loss is explained
# `here <https://distill.pub/2017/ctc/>`__. In CTC a blank token () is a
# special token which represents a repetition of the previous symbol. In
# decoding, these are simply ignored.
#


######################################################################
# Conclusion
# ----------
#
# In this tutorial, we looked at how to use :py:mod:`torchaudio.pipelines` to
# perform acoustic feature extraction and speech recognition. Constructing
# a model and getting the emission is as short as two lines.
#
# ::
#
#    model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
#    emission = model(waveforms, ...)
#


from . import (
    dataset,
    dist_utils,
    metrics,
)

__all__ = ['dataset', 'dist_utils', 'metrics']


from torchaudio_unittest.common_utils import sox_utils


def get_encoding(ext, dtype):
    exts = {
        'mp3',
        'flac',
        'vorbis',
    }
    encodings = {
        'float32': 'PCM_F',
        'int32': 'PCM_S',
        'int16': 'PCM_S',
        'uint8': 'PCM_U',
    }
    return ext.upper() if ext in exts else encodings[dtype]


def get_bits_per_sample(ext, dtype):
    bits_per_samples = {
        'flac': 24,
        'mp3': 0,
        'vorbis': 0,
    }
    return bits_per_samples.get(ext, sox_utils.get_bit_depth(dtype))


import time
from typing import Tuple
from collections import namedtuple

import torch
import torch.distributed as dist

from utils import dist_utils, metrics

_LG = dist_utils.getLogger(__name__)

Metric = namedtuple("SNR", ["si_snri", "sdri"])
Metric.__str__ = (
    lambda self: f"SI-SNRi: {self.si_snri:10.3e}, SDRi: {self.sdri:10.3e}"
)


def si_sdr_improvement(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mix: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the improvement of scale-invariant SDR. (SI-SNRi) and bare SDR (SDRi).

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Shape: [batch, speakers, time frame]
        reference (torch.Tensor): Reference (original) source signals.
            Shape: [batch, speakers, time frame]
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Shape: [batch, speakers == 1, time frame]
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame]


    Returns:
        torch.Tensor: Improved SI-SDR. Shape: [batch, ]
        torch.Tensor: Absolute SI-SDR. Shape: [batch, ]

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """
    with torch.no_grad():
        sdri = metrics.sdri(estimate, reference, mix, mask=mask)

    estimate = estimate - estimate.mean(axis=2, keepdim=True)
    reference = reference - reference.mean(axis=2, keepdim=True)
    mix = mix - mix.mean(axis=2, keepdim=True)

    si_sdri = metrics.sdri(estimate, reference, mix, mask=mask)
    return si_sdri, sdri


class OccasionalLogger:
    """Simple helper class to log once in a while or when progress is quick enough"""

    def __init__(self, time_interval=180, progress_interval=0.1):
        self.time_interval = time_interval
        self.progress_interval = progress_interval

        self.last_time = 0.0
        self.last_progress = 0.0

    def log(self, metric, progress, force=False):
        now = time.monotonic()
        if (
            force
            or now > self.last_time + self.time_interval
            or progress > self.last_progress + self.progress_interval
        ):
            self.last_time = now
            self.last_progress = progress
            _LG.info_on_master("train: %s [%3d%%]", metric, 100 * progress)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        valid_loader,
        eval_loader,
        grad_clip,
        device,
        *,
        debug,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader
        self.grad_clip = grad_clip
        self.device = device
        self.debug = debug

    def train_one_epoch(self):
        self.model.train()
        logger = OccasionalLogger()

        num_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader, start=1):
            mix = batch.mix.to(self.device)
            src = batch.src.to(self.device)
            mask = batch.mask.to(self.device)

            estimate = self.model(mix)

            si_snri, sdri = si_sdr_improvement(estimate, src, mix, mask)
            si_snri = si_snri.mean()
            sdri = sdri.mean()

            loss = -si_snri
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, norm_type=2.0
            )
            self.optimizer.step()

            metric = Metric(si_snri.item(), sdri.item())
            logger.log(metric, progress=i / num_batches, force=i == num_batches)

            if self.debug:
                break

    def evaluate(self):
        with torch.no_grad():
            return self._test(self.eval_loader)

    def validate(self):
        with torch.no_grad():
            return self._test(self.valid_loader)

    def _test(self, loader):
        self.model.eval()

        total_si_snri = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_sdri = torch.zeros(1, dtype=torch.float32, device=self.device)

        for batch in loader:
            mix = batch.mix.to(self.device)
            src = batch.src.to(self.device)
            mask = batch.mask.to(self.device)

            estimate = self.model(mix)

            si_snri, sdri = si_sdr_improvement(estimate, src, mix, mask)

            total_si_snri += si_snri.sum()
            total_sdri += sdri.sum()

            if self.debug:
                break

        dist.all_reduce(total_si_snri, dist.ReduceOp.SUM)
        dist.all_reduce(total_sdri, dist.ReduceOp.SUM)

        num_samples = len(loader.dataset)
        metric = Metric(total_si_snri.item() / num_samples, total_sdri.item() / num_samples)
        return metric




import collections
import itertools


class LanguageModel:
    def __init__(self, labels, char_blank, char_space):

        self.char_space = char_space
        self.char_blank = char_blank

        labels = list(labels)
        self.length = len(labels)
        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        d1 = collections.OrderedDict(enumerated)
        d2 = collections.OrderedDict(flipped)
        self.mapping = {**d1, **d2}

    def encode(self, iterable):
        if isinstance(iterable, list):
            return [self.encode(i) for i in iterable]
        else:
            return [self.mapping[i] + self.mapping[self.char_blank] for i in iterable]

    def decode(self, tensor):
        if len(tensor) > 0 and isinstance(tensor[0], list):
            return [self.decode(t) for t in tensor]
        else:
            # not idempotent, since clean string
            x = (self.mapping[i] for i in tensor)
            x = "".join(i for i, _ in itertools.groupby(x))
            x = x.replace(self.char_blank, "")
            # x = x.strip()
            return x

    def __len__(self):
        return self.length


import io
import os
import zipfile
import tarfile
import gzip
import shutil
from functools import partial

import torch.utils.data

from torchtext.data.utils import RandomShuffler
from .example import Example
from torchtext.utils import download_from_url, unicode_csv_reader


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for batching
            together examples with similar lengths to minimize padding.
        examples (list(Example)): The examples in this dataset.
        fields (dict[str, Field]): Contains the name of each column or field, together
            with the corresponding Field object. Two fields with the same Field object
            will have a shared vocabulary.
    """
    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        """Create Dataset objects for multiple splits of a dataset.

        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root (str): Root dataset storage directory. Default is '.data'.
            train (str): Suffix to add to path for the train set, or None for no
                train set. Default is None.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            Remaining keyword arguments: Passed to the constructor of the
                Dataset (sub)class being used.

        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if provided.
        """
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        """Create train-test(-valid?) splits from the instance's examples.

        Arguments:
            split_ratio (float or List of floats): a number [0, 1] denoting the amount
                of data to be used for the training split (rest is used for test),
                or a list of numbers denoting the relative sizes of train, test and valid
                splits respectively. If the relative size for valid is missing, only the
                train-test split is returned. Default is 0.7 (for the train set).
            stratified (bool): whether the sampling should be stratified.
                Default is False.
            strata_field (str): name of the examples Field stratified over.
                Default is 'label' for the conventional label field.
            random_state (tuple): the random seed used for shuffling.
                A return value of `random.getstate()`.

        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if the splits are provided.
        """
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)

        # For the permutations
        rnd = RandomShuffler(random_state)
        if not stratified:
            train_data, test_data, val_data = rationed_split(self.examples, train_ratio,
                                                             test_ratio, val_ratio, rnd)
        else:
            if strata_field not in self.fields:
                raise ValueError("Invalid field name for strata_field {}"
                                 .format(strata_field))
            strata = stratify(self.examples, strata_field)
            train_data, test_data, val_data = [], [], []
            for group in strata:
                # Stratify each group and add together the indices.
                group_train, group_test, group_val = rationed_split(group, train_ratio,
                                                                    test_ratio, val_ratio,
                                                                    rnd)
                train_data += group_train
                test_data += group_test
                val_data += group_val

        splits = tuple(Dataset(d, self.fields)
                       for d in (train_data, val_data, test_data) if d)

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    @classmethod
    def download(cls, root, check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).

        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.

        Returns:
            str: Path to extracted dataset.
        """
        path = os.path.join(root, cls.name)
        check = path if check is None else check
        if not os.path.isdir(check):
            for url in cls.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    download_from_url(url, zpath)
                zroot, ext = os.path.splitext(zpath)
                _, ext_inner = os.path.splitext(zroot)
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)
                # tarfile cannot handle bare .gz files
                elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
                elif ext == '.gz':
                    with gzip.open(zpath, 'rb') as gz:
                        with open(zroot, 'wb') as uncompressed:
                            shutil.copyfileobj(gz, uncompressed)

        return os.path.join(path, cls.dirname)

    def filter_examples(self, field_names):
        """Remove unknown words from dataset examples with respect to given field.

        Arguments:
            field_names (list(str)): Within example only the parts with field names in
                field_names will have their unknown words deleted.
        """
        for i, example in enumerate(self.examples):
            for field_name in field_names:
                vocab = set(self.fields[field_name].vocab.stoi)
                text = getattr(example, field_name)
                example_part = [word for word in text if word in vocab]
                setattr(example, field_name, example_part)
            self.examples[i] = example


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params=None, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.

        Args:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields ((list(tuple(str, Field)) or dict[str: tuple(str, Field)): If using a list,
                the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
            kwargs (dict): passed to the Dataset parent class.
        """
        if csv_reader_params is None:
            csv_reader_params = {}
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


def check_split_ratio(split_ratio):
    """Check that the split ratio argument is not malformed"""
    valid_ratio = 0.
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert 0. < split_ratio < 1., (
            "Split ratio {} not between 0 and 1".format(split_ratio))

        test_ratio = 1. - split_ratio
        return (split_ratio, test_ratio, valid_ratio)
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert length == 2 or length == 3, (
            "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError('Split ratio must be float or a list, got {}'
                         .format(type(split_ratio)))


def stratify(examples, strata_field):
    # The field has to be hashable otherwise this doesn't work
    # There's two iterations over the whole dataset here, which can be
    # reduced to just one if a dedicated method for stratified splitting is used
    unique_strata = set(getattr(example, strata_field) for example in examples)
    strata_maps = {s: [] for s in unique_strata}
    for example in examples:
        strata_maps[getattr(example, strata_field)].append(example)
    return list(strata_maps.values())


def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
    """Create a random permutation of examples, then split them by ratios

    Arguments:
        examples: a list of data
        train_ratio, test_ratio, val_ratio: split fractions.
        rnd: a random shuffler

    Examples:
        >>> examples = []
        >>> train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
        >>> rnd = torchtext.data.dataset.RandomShuffler(None)
        >>> train_examples, test_examples, valid_examples = \
                torchtext.data.dataset.rationed_split(examples, train_ratio,
                                                      test_ratio, val_ratio,
                                                      rnd)
    """
    N = len(examples)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (randperm[:train_len],  # Train
               randperm[train_len:train_len + test_len],  # Test
               randperm[train_len + test_len:])  # Validation

    # There's a possibly empty list for the validation set
    data = tuple([examples[i] for i in index] for index in indices)

    return data


import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime
from time import time
from typing import List

import torch
import torchaudio
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator
from torchaudio.models.wavernn import WaveRNN

from datasets import collate_factory, split_process_dataset
from losses import LongCrossEntropyLoss, MoLLoss
from processing import NormalizeDB
from utils import MetricLogger, count_parameters, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
    )
    parser.add_argument(
        "--epochs",
        default=8000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="manual epoch number"
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    parser.add_argument(
        "--dataset",
        default="ljspeech",
        choices=["ljspeech", "libritts"],
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--learning-rate", default=1e-4, type=float, metavar="LR", help="learning rate",
    )
    parser.add_argument("--clip-grad", metavar="NORM", type=float, default=4.0)
    parser.add_argument(
        "--mulaw",
        default=True,
        action="store_true",
        help="if used, waveform is mulaw encoded",
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="if used, model is jitted"
    )
    parser.add_argument(
        "--upsample-scales",
        default=[5, 5, 11],
        type=List[int],
        help="the list of upsample scales",
    )
    parser.add_argument(
        "--n-bits", default=8, type=int, help="the bits of output waveform",
    )
    parser.add_argument(
        "--sample-rate",
        default=22050,
        type=int,
        help="the rate of audio dimensions (samples per second)",
    )
    parser.add_argument(
        "--hop-length",
        default=275,
        type=int,
        help="the number of samples between the starts of consecutive frames",
    )
    parser.add_argument(
        "--win-length", default=1100, type=int, help="the length of the STFT window",
    )
    parser.add_argument(
        "--f-min", default=40.0, type=float, help="the minimum frequency",
    )
    parser.add_argument(
        "--min-level-db",
        default=-100,
        type=float,
        help="the minimum db value for spectrogam normalization",
    )
    parser.add_argument(
        "--n-res-block", default=10, type=int, help="the number of ResBlock in stack",
    )
    parser.add_argument(
        "--n-rnn", default=512, type=int, help="the dimension of RNN layer",
    )
    parser.add_argument(
        "--n-fc", default=512, type=int, help="the dimension of fully connected layer",
    )
    parser.add_argument(
        "--kernel-size",
        default=5,
        type=int,
        help="the number of kernel size in the first Conv1d layer",
    )
    parser.add_argument(
        "--n-freq", default=80, type=int, help="the number of spectrogram bins to use",
    )
    parser.add_argument(
        "--n-hidden-melresnet",
        default=128,
        type=int,
        help="the number of hidden dimensions of resblock in melresnet",
    )
    parser.add_argument(
        "--n-output-melresnet", default=128, type=int, help="the output dimension of melresnet",
    )
    parser.add_argument(
        "--n-fft", default=2048, type=int, help="the number of Fourier bins",
    )
    parser.add_argument(
        "--loss",
        default="crossentropy",
        choices=["crossentropy", "mol"],
        type=str,
        help="the type of loss",
    )
    parser.add_argument(
        "--seq-len-factor",
        default=5,
        type=int,
        help="the length of each waveform to process per batch = hop_length * seq_len_factor",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="the ratio of waveforms for validation",
    )
    parser.add_argument(
        "--file-path", default="", type=str, help="the path of audio files",
    )
    parser.add_argument(
        "--normalization", default=True, action="store_true", help="if True, spectrogram is normalized",
    )

    args = parser.parse_args()
    return args


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):

    model.train()

    sums = defaultdict(lambda: 0.0)
    start1 = time()

    metric = MetricLogger("train_iteration")
    metric["epoch"] = epoch

    for waveform, specgram, target in bg_iterator(data_loader, maxsize=2):

        start2 = time()

        waveform = waveform.to(device)
        specgram = specgram.to(device)
        target = target.to(device)

        output = model(waveform, specgram)
        output, target = output.squeeze(1), target.squeeze(1)

        loss = criterion(output, target)
        loss_item = loss.item()
        sums["loss"] += loss_item
        metric["loss"] = loss_item

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad > 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad
            )
            sums["gradient"] += gradient.item()
            metric["gradient"] = gradient.item()

        optimizer.step()

        metric["iteration"] = sums["iteration"]
        metric["time"] = time() - start2
        metric()
        sums["iteration"] += 1

    avg_loss = sums["loss"] / len(data_loader)

    metric = MetricLogger("train_epoch")
    metric["epoch"] = epoch
    metric["loss"] = sums["loss"] / len(data_loader)
    metric["gradient"] = avg_loss
    metric["time"] = time() - start1
    metric()


def validate(model, criterion, data_loader, device, epoch):

    with torch.no_grad():

        model.eval()
        sums = defaultdict(lambda: 0.0)
        start = time()

        for waveform, specgram, target in bg_iterator(data_loader, maxsize=2):

            waveform = waveform.to(device)
            specgram = specgram.to(device)
            target = target.to(device)

            output = model(waveform, specgram)
            output, target = output.squeeze(1), target.squeeze(1)

            loss = criterion(output, target)
            sums["loss"] += loss.item()

        avg_loss = sums["loss"] / len(data_loader)

        metric = MetricLogger("validation")
        metric["epoch"] = epoch
        metric["loss"] = avg_loss
        metric["time"] = time() - start
        metric()

        return avg_loss


def main(args):

    devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    logging.info("Start time: {}".format(str(datetime.now())))

    melkwargs = {
        "n_fft": args.n_fft,
        "power": 1,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
    }

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_mels=args.n_freq,
            f_min=args.f_min,
            mel_scale='slaney',
            norm='slaney',
            **melkwargs,
        ),
        NormalizeDB(min_level_db=args.min_level_db, normalization=args.normalization),
    )

    train_dataset, val_dataset = split_process_dataset(args, transforms)

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": False,
        "shuffle": True,
        "drop_last": False,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    collate_fn = collate_factory(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_training_params,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_validation_params,
    )

    n_classes = 2 ** args.n_bits if args.loss == "crossentropy" else 30

    model = WaveRNN(
        upsample_scales=args.upsample_scales,
        n_classes=n_classes,
        hop_length=args.hop_length,
        n_res_block=args.n_res_block,
        n_rnn=args.n_rnn,
        n_fc=args.n_fc,
        kernel_size=args.kernel_size,
        n_freq=args.n_freq,
        n_hidden=args.n_hidden_melresnet,
        n_output=args.n_output_melresnet,
    )

    if args.jit:
        model = torch.jit.script(model)

    model = torch.nn.DataParallel(model)
    model = model.to(devices[0], non_blocking=True)

    n = count_parameters(model)
    logging.info(f"Number of parameters: {n}")

    # Optimizer
    optimizer_params = {
        "lr": args.learning_rate,
    }

    optimizer = Adam(model.parameters(), **optimizer_params)

    criterion = LongCrossEntropyLoss() if args.loss == "crossentropy" else MoLLoss()

    best_loss = 10.0

    if args.checkpoint and os.path.isfile(args.checkpoint):
        logging.info(f"Checkpoint: loading '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        logging.info(
            f"Checkpoint: loaded '{args.checkpoint}' at epoch {checkpoint['epoch']}"
        )
    else:
        logging.info("Checkpoint: not found")

        save_checkpoint(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            False,
            args.checkpoint,
        )

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model, criterion, optimizer, train_loader, devices[0], epoch,
        )

        if not (epoch + 1) % args.print_freq or epoch == args.epochs - 1:

            sum_loss = validate(model, criterion, val_loader, devices[0], epoch)

            is_best = sum_loss < best_loss
            best_loss = min(sum_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.checkpoint,
            )

    logging.info(f"End time: {datetime.now()}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)


from typing import Optional

import torch
import scipy.io.wavfile


def normalize_wav(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        pass
    elif tensor.dtype == torch.int32:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 2147483647.
        tensor[tensor < 0] /= 2147483648.
    elif tensor.dtype == torch.int16:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 32767.
        tensor[tensor < 0] /= 32768.
    elif tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) - 128
        tensor[tensor > 0] /= 127.
        tensor[tensor < 0] /= 128.
    return tensor


def get_wav_data(
        dtype: str,
        num_channels: int,
        *,
        num_frames: Optional[int] = None,
        normalize: bool = True,
        channels_first: bool = True,
):
    """Generate linear signal of the given dtype and num_channels

    Data range is
        [-1.0, 1.0] for float32,
        [-2147483648, 2147483647] for int32
        [-32768, 32767] for int16
        [0, 255] for uint8

    num_frames allow to change the linear interpolation parameter.
    Default values are 256 for uint8, else 1 << 16.
    1 << 16 as default is so that int16 value range is completely covered.
    """
    dtype_ = getattr(torch, dtype)

    if num_frames is None:
        if dtype == 'uint8':
            num_frames = 256
        else:
            num_frames = 1 << 16

    if dtype == 'uint8':
        base = torch.linspace(0, 255, num_frames, dtype=dtype_)
    elif dtype == 'int8':
        base = torch.linspace(-128, 127, num_frames, dtype=dtype_)
    elif dtype == 'float32':
        base = torch.linspace(-1., 1., num_frames, dtype=dtype_)
    elif dtype == 'float64':
        base = torch.linspace(-1., 1., num_frames, dtype=dtype_)
    elif dtype == 'int32':
        base = torch.linspace(-2147483648, 2147483647, num_frames, dtype=dtype_)
    elif dtype == 'int16':
        base = torch.linspace(-32768, 32767, num_frames, dtype=dtype_)
    else:
        raise NotImplementedError(f'Unsupported dtype {dtype}')
    data = base.repeat([num_channels, 1])
    if not channels_first:
        data = data.transpose(1, 0)
    if normalize:
        data = normalize_wav(data)
    return data


def load_wav(path: str, normalize=True, channels_first=True) -> torch.Tensor:
    """Load wav file without torchaudio"""
    sample_rate, data = scipy.io.wavfile.read(path)
    data = torch.from_numpy(data.copy())
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if normalize:
        data = normalize_wav(data)
    if channels_first:
        data = data.transpose(1, 0)
    return data, sample_rate


def save_wav(path, data, sample_rate, channels_first=True):
    """Save wav file without torchaudio"""
    if channels_first:
        data = data.transpose(1, 0)
    scipy.io.wavfile.write(path, sample_rate, data.numpy())


import argparse
import time
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from model import NextSentenceTask, BertModel, BertEmbedding
from utils import run_demo, run_ddp, wrap_up


def process_raw_data(whole_data, args):
    processed_data = []
    for _idx in range(len(whole_data)):
        item = whole_data[_idx]
        if isinstance(item, list):
            item = torch.tensor(item)
        if len(item) > 1:
            # idx to split the text into two sentencd
            split_idx = torch.randint(1, len(item), size=(1, 1)).item()
            # Index 2 means same sentence label. Initial true int(1)
            processed_data.append([item[:split_idx], item[split_idx:], 1])
    # Random shuffle data to have args.frac_ns next sentence set up
    shuffle_idx1 = torch.randperm(len(processed_data))
    shuffle_idx2 = torch.randperm(len(processed_data))
    num_shuffle = int(len(processed_data) * args.frac_ns)
    shuffle_zip = list(zip(shuffle_idx1, shuffle_idx2))[:num_shuffle]
    for (i, j) in shuffle_zip:
        processed_data[i][1] = processed_data[j][0]
        processed_data[i][2] = int(0)  # Switch same sentence label to false 0
    return processed_data


def collate_batch(batch, args, cls_id, sep_id, pad_id):
    # Fix sequence length to args.bptt with padding or trim
    seq_list = []
    tok_type = []
    same_sentence_labels = []
    for item in batch:
        qa_item = torch.cat([item[0], torch.tensor([sep_id]).long(), item[1], torch.tensor([sep_id]).long()])
        if qa_item.size(0) > args.bptt:
            qa_item = qa_item[:args.bptt]
        elif qa_item.size(0) < args.bptt:
            qa_item = torch.cat((qa_item,
                                 torch.tensor([pad_id] * (args.bptt -
                                              qa_item.size(0)))))
        seq_list.append(qa_item)
        _tok_tp = torch.ones((qa_item.size(0)))
        _idx = min(len(item[0]) + 1, args.bptt)
        _tok_tp[:_idx] = 0.0
        tok_type.append(_tok_tp)
        same_sentence_labels.append(item[2])
    seq_input = torch.stack(seq_list).long().t().contiguous()
    seq_input = torch.cat((torch.tensor([[cls_id] * seq_input.size(1)]).long(), seq_input))
    tok_type = torch.stack(tok_type).long().t().contiguous()
    tok_type = torch.cat((torch.tensor([[0] * tok_type.size(1)]).long(), tok_type))
    return seq_input, tok_type, torch.tensor(same_sentence_labels).long().contiguous()


def evaluate(data_source, model, device, criterion, cls_id, sep_id, pad_id, args):
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_batch(b, args, cls_id, sep_id, pad_id))
    with torch.no_grad():
        for idx, (seq_input, tok_type, target_ns_labels) in enumerate(dataloader):
            if args.parallel == 'DDP':
                seq_input = seq_input.to(device[0])
                tok_type = tok_type.to(device[0])
                target_ns_labels = target_ns_labels.to(device[0])
            else:
                seq_input = seq_input.to(device)
                tok_type = tok_type.to(device)
                target_ns_labels = target_ns_labels.to(device)
            seq_input = seq_input.transpose(0, 1)  # Wrap up by DDP or DataParallel
            ns_labels = model(seq_input, token_type_input=tok_type)
            loss = criterion(ns_labels, target_ns_labels)
            total_loss += loss.item()
    return total_loss / (len(data_source) // batch_size)


def train(train_dataset, model, train_loss_log, device, optimizer, criterion,
          epoch, scheduler, cls_id, sep_id, pad_id, args, rank=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_batch(b, args, cls_id, sep_id, pad_id))
    train_loss_log.append(0.0)
    for idx, (seq_input, tok_type, target_ns_labels) in enumerate(dataloader):
        if args.parallel == 'DDP':
            seq_input = seq_input.to(device[0])
            tok_type = tok_type.to(device[0])
            target_ns_labels = target_ns_labels.to(device[0])
        else:
            seq_input = seq_input.to(device)
            tok_type = tok_type.to(device)
            target_ns_labels = target_ns_labels.to(device)
        optimizer.zero_grad()
        seq_input = seq_input.transpose(0, 1)  # Wrap up by DDP or DataParallel
        ns_labels = model(seq_input, token_type_input=tok_type)
        loss = criterion(ns_labels, target_ns_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if (rank is None) or rank == 0:
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                      'ms/batch {:5.2f} | '
                      'loss {:8.5f} | ppl {:5.2f}'.format(epoch, idx,
                                                          len(train_dataset) // batch_size,
                                                          scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / args.log_interval,
                                                          cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def run_main(args, rank=None):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if args.parallel == 'DDP':
        n = torch.cuda.device_count() // args.world_size
        device = list(range(rank * n, (rank + 1) * n))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = torch.load(args.save_vocab)
    cls_id = vocab.stoi['<cls>']
    pad_id = vocab.stoi['<pad>']
    sep_id = vocab.stoi['<sep>']

    if args.dataset == 'WikiText103':
        from torchtext.experimental.datasets import WikiText103
        train_dataset, valid_dataset, test_dataset = WikiText103(vocab=vocab)
    elif args.dataset == 'BookCorpus':
        from data import BookCorpus
        train_dataset, valid_dataset, test_dataset = BookCorpus(vocab, min_sentence_len=60)

    if rank is not None:
        chunk_len = len(train_dataset.data) // args.world_size
        train_dataset.data = train_dataset.data[(rank * chunk_len):((rank + 1) * chunk_len)]

    if args.checkpoint != 'None':
        model = torch.load(args.checkpoint)
    else:
        embed_layer = BertEmbedding(len(vocab), args.emsize)
        pretrained_bert = BertModel(len(vocab), args.emsize, args.nhead, args.nhid, args.nlayers, embed_layer, args.dropout)
        pretrained_bert.load_state_dict(torch.load(args.bert_model))
        model = NextSentenceTask(pretrained_bert)

    if args.parallel == 'DDP':
        model = model.to(device[0])
        model = DDP(model, device_ids=device)
    else:
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(process_raw_data(train_dataset, args), model, train_loss_log, device, optimizer,
              criterion, epoch, scheduler, cls_id, sep_id, pad_id, args, rank)
        val_loss = evaluate(process_raw_data(valid_dataset, args), model, device, criterion,
                            cls_id, sep_id, pad_id, args)
        val_loss_log.append(val_loss)

        if (rank is None) or (rank == 0):
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| valid loss {:8.5f} | '.format(epoch,
                                                   (time.time() - epoch_start_time),
                                                   val_loss))
            print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            if rank is None:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            elif rank == 0:
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            scheduler.step()
    if args.parallel == 'DDP':
        rank0_devices = [x - rank * len(device) for x in device]
        device_pairs = zip(rank0_devices, device)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        model.load_state_dict(torch.load(args.save, map_location=map_location))
        test_loss = evaluate(process_raw_data(test_dataset, args), model, device, criterion,
                             cls_id, sep_id, pad_id, args)
        if rank == 0:
            wrap_up(train_loss_log, val_loss_log, test_loss, args, model.module, 'ns_loss.txt', 'ns_model.pt')
    else:
        with open(args.save, 'rb') as f:
            model = torch.load(f)

        test_loss = evaluate(process_raw_data(test_dataset, args), model, device, criterion,
                             cls_id, sep_id, pad_id, args)
        wrap_up(train_loss_log, val_loss_log, test_loss, args, model, 'ns_loss.txt', 'ns_model.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question-Answer fine-tuning task')
    parser.add_argument('--dataset', type=str, default='WikiText103',
                        help='dataset used for next sentence task')
    parser.add_argument('--lr', type=float, default=0.25,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='max. sequence length for the next-sentence pair')
    parser.add_argument('--min_sentence_len', type=int, default=60,
                        help='min. sequence length for the raw text tokens')
    parser.add_argument('--seed', type=int, default=312216194,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=600, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    parser.add_argument('--save', type=str, default='ns_bert.pt',
                        help='path to save the bert model')
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str, default='mlm_bert.pt',
                        help='path to save the pretrained bert')
    parser.add_argument('--frac_ns', type=float, default=0.5,
                        help='fraction of not next sentence')
    parser.add_argument('--parallel', type=str, default='None',
                        help='Use DataParallel/DDP to train model')
    parser.add_argument('--world_size', type=int, default=8,
                        help='the world size to initiate DPP')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    args = parser.parse_args()

    if args.parallel == 'DDP':
        run_demo(run_ddp, run_main, args)
    else:
        run_main(args)


import os
import json

from transformers import Wav2Vec2Model

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _main():
    keys = [
        # pretrained
        "facebook/wav2vec2-base",
        "facebook/wav2vec2-large",
        "facebook/wav2vec2-large-lv60",
        "facebook/wav2vec2-base-10k-voxpopuli",
        "facebook/wav2vec2-large-xlsr-53",
        # finetuned
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h",
        "facebook/wav2vec2-large-960h-lv60",
        "facebook/wav2vec2-large-960h-lv60-self",
        "facebook/wav2vec2-large-xlsr-53-german",
    ]
    for key in keys:
        path = os.path.join(_THIS_DIR, f'{key}.json')
        print('Generating ', path)
        cfg = Wav2Vec2Model.from_pretrained(key).config
        cfg = json.loads(cfg.to_json_string())
        del cfg['_name_or_path']

        with open(path, 'w') as file_:
            file_.write(json.dumps(cfg, indent=4, sort_keys=True))
            file_.write('\n')


if __name__ == '__main__':
    _main()


from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from torch import Tensor


def load(filepath: Union[str, Path],
         out: Optional[Tensor] = None,
         normalization: Union[bool, float, Callable] = True,
         channels_first: bool = True,
         num_frames: int = 0,
         offset: int = 0,
         filetype: Optional[str] = None) -> Tuple[Tensor, int]:
    raise RuntimeError('No audio I/O backend is available.')


def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    raise RuntimeError('No audio I/O backend is available.')


def info(filepath: str) -> None:
    raise RuntimeError('No audio I/O backend is available.')


import unittest
import random
import torch
import numpy as np
from torchaudio.functional import rnnt_loss


CPU_DEVICE = torch.device("cpu")


class _NumpyTransducer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        logit_lengths,
        target_lengths,
        targets,
        blank=-1,
    ):
        device = log_probs.device
        log_probs = log_probs.cpu().data.numpy()
        logit_lengths = logit_lengths.cpu().data.numpy()
        target_lengths = target_lengths.cpu().data.numpy()
        targets = targets.cpu().data.numpy()

        gradients, costs, _, _ = __class__.compute(
            log_probs=log_probs,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            targets=targets,
            blank=blank,
        )

        costs = torch.FloatTensor(costs).to(device=device)
        gradients = torch.FloatTensor(gradients).to(device=device)
        ctx.grads = torch.autograd.Variable(gradients)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul(grad_output), None, None, None, None, None, None, None, None

    @staticmethod
    def compute_alpha_one_sequence(log_probs, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        alpha = np.zeros((max_T, max_U), dtype=np.float32)
        for t in range(1, max_T):
            alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank]

        for u in range(1, max_U):
            alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, targets[u - 1]]

        for t in range(1, max_T):
            for u in range(1, max_U):
                skip = alpha[t - 1, u] + log_probs[t - 1, u, blank]
                emit = alpha[t, u - 1] + log_probs[t, u - 1, targets[u - 1]]
                alpha[t, u] = np.logaddexp(skip, emit)

        cost = -(alpha[-1, -1] + log_probs[-1, -1, blank])
        return alpha, cost

    @staticmethod
    def compute_beta_one_sequence(log_probs, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        beta = np.zeros((max_T, max_U), dtype=np.float32)
        beta[-1, -1] = log_probs[-1, -1, blank]

        for t in reversed(range(max_T - 1)):
            beta[t, -1] = beta[t + 1, -1] + log_probs[t, -1, blank]

        for u in reversed(range(max_U - 1)):
            beta[-1, u] = beta[-1, u + 1] + log_probs[-1, u, targets[u]]

        for t in reversed(range(max_T - 1)):
            for u in reversed(range(max_U - 1)):
                skip = beta[t + 1, u] + log_probs[t, u, blank]
                emit = beta[t, u + 1] + log_probs[t, u, targets[u]]
                beta[t, u] = np.logaddexp(skip, emit)

        cost = -beta[0, 0]
        return beta, cost

    @staticmethod
    def compute_gradients_one_sequence(
        log_probs, alpha, beta, targets, blank=-1
    ):
        max_T, max_U, D = log_probs.shape
        gradients = np.full(log_probs.shape, float("-inf"))
        cost = -beta[0, 0]

        gradients[-1, -1, blank] = alpha[-1, -1]

        gradients[:-1, :, blank] = alpha[:-1, :] + beta[1:, :]

        for u, l in enumerate(targets):
            gradients[:, u, l] = alpha[:, u] + beta[:, u + 1]

        gradients = -(np.exp(gradients + log_probs + cost))
        return gradients

    @staticmethod
    def compute(
        log_probs,
        logit_lengths,
        target_lengths,
        targets,
        blank=-1,
    ):
        gradients = np.zeros_like(log_probs)
        B_tgt, max_T, max_U, D = log_probs.shape
        B_src = logit_lengths.shape[0]

        H = int(B_tgt / B_src)

        alphas = np.zeros((B_tgt, max_T, max_U))
        betas = np.zeros((B_tgt, max_T, max_U))
        betas.fill(float("-inf"))
        alphas.fill(float("-inf"))
        costs = np.zeros(B_tgt)
        for b_tgt in range(B_tgt):
            b_src = int(b_tgt / H)
            T = int(logit_lengths[b_src])
            # NOTE: see https://arxiv.org/pdf/1211.3711.pdf Section 2.1
            U = int(target_lengths[b_tgt]) + 1

            seq_log_probs = log_probs[b_tgt, :T, :U, :]
            seq_targets = targets[b_tgt, : int(target_lengths[b_tgt])]
            alpha, alpha_cost = __class__.compute_alpha_one_sequence(
                log_probs=seq_log_probs, targets=seq_targets, blank=blank
            )

            beta, beta_cost = __class__.compute_beta_one_sequence(
                log_probs=seq_log_probs, targets=seq_targets, blank=blank
            )

            seq_gradients = __class__.compute_gradients_one_sequence(
                log_probs=seq_log_probs,
                alpha=alpha,
                beta=beta,
                targets=seq_targets,
                blank=blank,
            )
            np.testing.assert_almost_equal(alpha_cost, beta_cost, decimal=2)
            gradients[b_tgt, :T, :U, :] = seq_gradients
            costs[b_tgt] = beta_cost
            alphas[b_tgt, :T, :U] = alpha
            betas[b_tgt, :T, :U] = beta

        return gradients, costs, alphas, betas


class NumpyTransducerLoss(torch.nn.Module):
    def __init__(self, blank=-1):
        super().__init__()
        self.blank = blank

    def forward(
        self,
        logits,
        logit_lengths,
        target_lengths,
        targets,
    ):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return _NumpyTransducer.apply(
            log_probs,
            logit_lengths,
            target_lengths,
            targets,
            self.blank,
        )


def compute_with_numpy_transducer(data):
    costs = NumpyTransducerLoss(
        blank=data["blank"],
    )(
        logits=data["logits"],
        logit_lengths=data["logit_lengths"],
        target_lengths=data["target_lengths"],
        targets=data["targets"],
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu()
    gradients = data["logits"].saved_grad.cpu()
    return costs, gradients


def compute_with_pytorch_transducer(data):
    costs = rnnt_loss(
        logits=data["logits"],
        logit_lengths=data["logit_lengths"],
        target_lengths=data["target_lengths"],
        targets=data["targets"],
        blank=data["blank"],
        reduction="none",
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu()
    gradients = data["logits"].saved_grad.cpu()
    return costs, gradients


def get_basic_data(device):
    # Example provided
    # in 6f73a2513dc784c59eec153a45f40bc528355b18
    # of https://github.com/HawkAaron/warp-transducer

    logits = torch.tensor(
        [
            [
                [
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.6, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.8, 0.1],
                ],
                [
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.1],
                    [0.7, 0.1, 0.2, 0.1, 0.1],
                ],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor([[1, 2]], dtype=torch.int, device=device)
    logit_lengths = torch.tensor([2], dtype=torch.int, device=device)
    target_lengths = torch.tensor([2], dtype=torch.int, device=device)

    logits.requires_grad_(True)

    return logits, targets, logit_lengths, target_lengths


def get_B1_T10_U3_D4_data(
    random=False,
    dtype=torch.float32,
    device=CPU_DEVICE,
):
    B, T, U, D = 2, 10, 3, 4

    logits = torch.rand(B, T, U, D, dtype=dtype, device=device)
    if not random:
        logits.fill_(0.1)
    logits.requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()
    logits.register_hook(grad_hook)

    data = {}
    data["logits"] = logits
    data["logit_lengths"] = torch.tensor([10, 10], dtype=torch.int32, device=device)
    data["target_lengths"] = torch.tensor([2, 2], dtype=torch.int32, device=device)
    data["targets"] = torch.tensor([[1, 2], [1, 2]], dtype=torch.int32, device=device)
    data["blank"] = 0

    return data


def get_B1_T2_U3_D5_data(dtype=torch.float32, device=CPU_DEVICE):
    logits = torch.tensor(
        [
            0.1,
            0.6,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.6,
            0.1,
            0.1,
            0.1,
            0.1,
            0.2,
            0.8,
            0.1,
            0.1,
            0.6,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.2,
            0.1,
            0.1,
            0.7,
            0.1,
            0.2,
            0.1,
            0.1,
        ],
        dtype=dtype,
        device=device,
    ).reshape(1, 2, 3, 5)
    logits.requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()
    logits.register_hook(grad_hook)

    targets = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    logit_lengths = torch.tensor([2], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([2], dtype=torch.int32, device=device)

    blank = -1

    ref_costs = torch.tensor([5.09566688538], dtype=dtype)
    ref_gradients = torch.tensor(
        [
            0.17703132,
            -0.39992708,
            0.17703132,
            0.17703132,
            -0.13116692,
            0.12247062,
            0.12247062,
            -0.181684,
            0.12247062,
            -0.1857276,
            0.06269141,
            0.06269141,
            0.06928471,
            0.12624498,
            -0.32091248,
            0.05456069,
            -0.2182428,
            0.05456069,
            0.05456069,
            0.05456069,
            0.12073967,
            0.12073967,
            -0.48295838,
            0.12073967,
            0.12073967,
            0.30741188,
            0.16871123,
            0.18645471,
            0.16871123,
            -0.83128875,
        ],
        dtype=dtype,
    ).reshape(1, 2, 3, 5)

    data = {
        "logits": logits,
        "targets": targets,
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def get_B2_T4_U3_D3_data(dtype=torch.float32, device=CPU_DEVICE):
    # Test from D21322854
    logits = torch.tensor(
        [
            0.065357,
            0.787530,
            0.081592,
            0.529716,
            0.750675,
            0.754135,
            0.609764,
            0.868140,
            0.622532,
            0.668522,
            0.858039,
            0.164539,
            0.989780,
            0.944298,
            0.603168,
            0.946783,
            0.666203,
            0.286882,
            0.094184,
            0.366674,
            0.736168,
            0.166680,
            0.714154,
            0.399400,
            0.535982,
            0.291821,
            0.612642,
            0.324241,
            0.800764,
            0.524106,
            0.779195,
            0.183314,
            0.113745,
            0.240222,
            0.339470,
            0.134160,
            0.505562,
            0.051597,
            0.640290,
            0.430733,
            0.829473,
            0.177467,
            0.320700,
            0.042883,
            0.302803,
            0.675178,
            0.569537,
            0.558474,
            0.083132,
            0.060165,
            0.107958,
            0.748615,
            0.943918,
            0.486356,
            0.418199,
            0.652408,
            0.024243,
            0.134582,
            0.366342,
            0.295830,
            0.923670,
            0.689929,
            0.741898,
            0.250005,
            0.603430,
            0.987289,
            0.592606,
            0.884672,
            0.543450,
            0.660770,
            0.377128,
            0.358021,
        ],
        dtype=dtype,
        device=device,
    ).reshape(2, 4, 3, 3)
    logits.requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()
    logits.register_hook(grad_hook)

    targets = torch.tensor([[1, 2], [1, 1]], dtype=torch.int32, device=device)
    logit_lengths = torch.tensor([4, 4], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([2, 2], dtype=torch.int32, device=device)

    blank = 0

    ref_costs = torch.tensor([4.2806528590890736, 3.9384369822503591], dtype=dtype)

    ref_gradients = torch.tensor(
        [
            -0.186844,
            -0.062555,
            0.249399,
            -0.203377,
            0.202399,
            0.000977,
            -0.141016,
            0.079123,
            0.061893,
            -0.011552,
            -0.081280,
            0.092832,
            -0.154257,
            0.229433,
            -0.075176,
            -0.246593,
            0.146405,
            0.100188,
            -0.012918,
            -0.061593,
            0.074512,
            -0.055986,
            0.219831,
            -0.163845,
            -0.497627,
            0.209240,
            0.288387,
            0.013605,
            -0.030220,
            0.016615,
            0.113925,
            0.062781,
            -0.176706,
            -0.667078,
            0.367659,
            0.299419,
            -0.356344,
            -0.055347,
            0.411691,
            -0.096922,
            0.029459,
            0.067463,
            -0.063518,
            0.027654,
            0.035863,
            -0.154499,
            -0.073942,
            0.228441,
            -0.166790,
            -0.000088,
            0.166878,
            -0.172370,
            0.105565,
            0.066804,
            0.023875,
            -0.118256,
            0.094381,
            -0.104707,
            -0.108934,
            0.213642,
            -0.369844,
            0.180118,
            0.189726,
            0.025714,
            -0.079462,
            0.053748,
            0.122328,
            -0.238789,
            0.116460,
            -0.598687,
            0.302203,
            0.296484,
        ],
        dtype=dtype,
    ).reshape(2, 4, 3, 3)

    data = {
        "logits": logits,
        "targets": targets,
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def get_random_data(
    max_B=8,
    max_T=128,
    max_U=32,
    max_D=40,
    blank=-1,
    dtype=torch.float32,
    device=CPU_DEVICE,
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed=seed)

    if blank != -1:
        raise ValueError("blank != -1 is not supported yet.")

    random.seed(0)
    B = random.randint(1, max_B - 1)
    T = random.randint(5, max_T - 1)
    U = random.randint(5, max_U - 1)
    D = random.randint(2, max_D - 1)

    logit_lengths = torch.randint(low=5, high=T + 1, size=(B,), dtype=torch.int32, device=device)
    target_lengths = torch.randint(low=5, high=U + 1, size=(B,), dtype=torch.int32, device=device)
    max_src_length = torch.max(logit_lengths)
    max_tgt_length = torch.max(target_lengths)

    targets = torch.randint(
        low=0, high=D - 1, size=(B, max_tgt_length), dtype=torch.int32, device=device
    )
    logits = torch.rand(
        size=(B, max_src_length, max_tgt_length + 1, D),
        dtype=dtype,
        device=device,
    ).requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()
    logits.register_hook(grad_hook)

    return {
        "logits": logits,
        "targets": targets,
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }


def skipIfNoRNNT(test_item):
    try:
        torch.ops.torchaudio.rnnt_loss
        return test_item
    except RuntimeError:
        return unittest.skip("torchaudio C++ extension is not compiled with RNN transducer loss")


import importlib
from .wmtnewscrawl import WMTNewsCrawl
from .wmt14 import WMT14

DATASETS = {
    'WMTNewsCrawl': WMTNewsCrawl,
    'WMT14': WMT14,
}

URLS = {}
NUM_LINES = {}
MD5 = {}
for dataset in DATASETS:
    dataset_module_path = "torchtext.experimental.datasets.raw." + dataset.lower()
    dataset_module = importlib.import_module(dataset_module_path)
    URLS[dataset] = dataset_module.URL
    NUM_LINES[dataset] = dataset_module.NUM_LINES
    MD5[dataset] = dataset_module.MD5

__all__ = sorted(list(map(str, DATASETS.keys())))


#!/usr/bin/env python3
"""Tests that requires external resources (Network access to fetch dataset)"""
import os
import unittest
from collections import Counter

import torch
import torchtext.data

from .common.torchtext_test_case import TorchtextTestCase


class TestNestedField(TorchtextTestCase):
    def test_build_vocab(self):
        nesting_field = torchtext.legacy.data.Field(tokenize=list, init_token="<w>", eos_token="</w>")

        field = torchtext.legacy.data.NestedField(
            nesting_field, init_token='<s>', eos_token='</s>',
            include_lengths=True,
            pad_first=True)

        sources = [
            [['a'], ['s', 'e', 'n', 't', 'e', 'n', 'c', 'e'], ['o', 'f'], ['d', 'a', 't', 'a'], ['.']],
            [['y', 'e', 't'], ['a', 'n', 'o', 't', 'h', 'e', 'r']],
            [['o', 'n', 'e'], ['l', 'a', 's', 't'], ['s', 'e', 'n', 't']]
        ]

        field.build_vocab(
            sources, vectors='glove.6B.50d',
            unk_init=torch.nn.init.normal_, vectors_cache=".vector_cache")


class TestDataset(TorchtextTestCase):
    def test_csv_file_no_header_one_col_multiple_fields(self):
        self.write_test_ppid_dataset(data_format="csv")

        question_field = torchtext.legacy.data.Field(sequential=True)
        spacy_tok_question_field = torchtext.legacy.data.Field(sequential=True, tokenize="spacy")
        label_field = torchtext.legacy.data.Field(sequential=False)
        # Field name/value as nested tuples
        fields = [("ids", None),
                  (("q1", "q1_spacy"), (question_field, spacy_tok_question_field)),
                  (("q2", "q2_spacy"), (question_field, spacy_tok_question_field)),
                  ("label", label_field)]
        dataset = torchtext.legacy.data.TabularDataset(
            path=self.test_ppid_dataset_path, format="csv", fields=fields)
        expected_examples = [
            (["When", "do", "you", "use", "", "instead", "of", "?"],
             ["When", "do", "you", "use", "", "instead", "of", "", "?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"],
             ["When", "do", "you", "use", "\"", "&", "\"",
              "instead", "of", "\"", "and", "\"", "?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Where", "was", "Lincoln", "born", "?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born", "?"],
             "1"),
            (["What", "is", "2+2"], ["What", "is", "2", "+", "2"],
             ["2+2=?"], ["2", "+", "2=", "?"], "1")]
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q1_spacy, expected_examples[i][1])
            self.assertEqual(example.q2, expected_examples[i][2])
            self.assertEqual(example.q2_spacy, expected_examples[i][3])
            self.assertEqual(example.label, expected_examples[i][4])

        # 6 Fields including None for ids
        assert len(dataset.fields) == 6

    def test_json_dataset_one_key_multiple_fields(self):
        self.write_test_ppid_dataset(data_format="json")

        question_field = torchtext.legacy.data.Field(sequential=True)
        spacy_tok_question_field = torchtext.legacy.data.Field(sequential=True, tokenize="spacy")
        label_field = torchtext.legacy.data.Field(sequential=False)
        fields = {"question1": [("q1", question_field),
                                ("q1_spacy", spacy_tok_question_field)],
                  "question2": [("q2", question_field),
                                ("q2_spacy", spacy_tok_question_field)],
                  "label": ("label", label_field)}
        dataset = torchtext.legacy.data.TabularDataset(
            path=self.test_ppid_dataset_path, format="json", fields=fields)
        expected_examples = [
            (["When", "do", "you", "use", "", "instead", "of", "?"],
             ["When", "do", "you", "use", "", "instead", "of", "", "?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"],
             ["When", "do", "you", "use", "\"", "&", "\"",
              "instead", "of", "\"", "and", "\"", "?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Where", "was", "Lincoln", "born", "?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born", "?"],
             "1"),
            (["What", "is", "2+2"], ["What", "is", "2", "+", "2"],
             ["2+2=?"], ["2", "+", "2=", "?"], "1")]
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q1_spacy, expected_examples[i][1])
            self.assertEqual(example.q2, expected_examples[i][2])
            self.assertEqual(example.q2_spacy, expected_examples[i][3])
            self.assertEqual(example.label, expected_examples[i][4])


class TestDataUtils(TorchtextTestCase):
    TEST_STR = "A string, particularly one with slightly complex punctuation."

    def test_get_tokenizer_spacy(self):
        # Test SpaCy option, and verify it properly handles punctuation.
        assert torchtext.data.get_tokenizer("spacy", language='en_core_web_sm')(str(self.TEST_STR)) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

    def test_get_tokenizer_moses(self):
        # Test Moses option.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        moses_tokenizer = torchtext.data.get_tokenizer("moses")
        assert moses_tokenizer(self.TEST_STR) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Nonbreaking prefixes should tokenize the final period.
        assert moses_tokenizer("abc def.") == ["abc", "def", "."]


class TestVocab(TorchtextTestCase):
    def test_vectors_get_vecs(self):
        vec = torchtext.vocab.GloVe(name='twitter.27B', dim='25')
        self.assertEqual(vec.vectors.shape[0], len(vec))

        tokens = ['chip', 'baby', 'Beautiful']
        token_vecs = vec.get_vecs_by_tokens(tokens)
        self.assertEqual(token_vecs.shape[0], len(tokens))
        self.assertEqual(token_vecs.shape[1], vec.dim)
        self.assertEqual(vec[tokens[0]], token_vecs[0])
        self.assertEqual(vec[tokens[1]], token_vecs[1])
        self.assertEqual(vec['<unk>'], token_vecs[2])

        token_one_vec = vec.get_vecs_by_tokens(tokens[0], lower_case_backup=True)
        self.assertEqual(token_one_vec.shape[0], vec.dim)
        self.assertEqual(vec[tokens[0].lower()], token_one_vec)

    def test_download_charngram_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'IO_TT': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "charngram.100d"
            else:
                vectors = torchtext.vocab.CharNGram()
            v = torchtext.legacy.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=vectors)
            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'IO_TT', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_charngram = {
                'hello': [-0.44782442, -0.08937783, -0.34227219,
                          -0.16233221, -0.39343098],
                'world': [-0.29590717, -0.05275926, -0.37334684, 0.27117205, -0.3868292],
            }

            for word in expected_charngram:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_charngram[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(100))
            self.assertEqual(vectors[v.stoi['OOV token']], torch.zeros(100))

    def test_download_custom_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'IO_TT': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            v = torchtext.legacy.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                vectors=torchtext.vocab.Vectors(
                    'wiki.simple.vec',
                    url=torchtext.vocab.FastText.url_base.format('simple')
                )
            )

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'IO_TT', 'hello', 'world'])
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))

    def test_download_fasttext_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'IO_TT': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "fasttext.simple.300d"
            else:
                vectors = torchtext.vocab.FastText(language='simple')

            v = torchtext.legacy.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=vectors)

            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'IO_TT', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))
            self.assertEqual(vectors[v.stoi['OOV token']], torch.zeros(300))

    def test_download_glove_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'IO_TT': 5, 'freq_too_low': 2})

        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "glove.twitter.27B.25d"
            else:
                vectors = torchtext.vocab.GloVe(name='twitter.27B', dim='25')
            v = torchtext.legacy.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=vectors)

            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'IO_TT', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)

            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_twitter = {
                'hello': [-0.77069, 0.12827, 0.33137, 0.0050893, -0.47605],
                'world': [0.10301, 0.095666, -0.14789, -0.22383, -0.14775],
            }

            for word in expected_twitter:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_twitter[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(25))
            self.assertEqual(vectors[v.stoi['OOV token']], torch.zeros(25))

    def test_extend(self):
        c = Counter({'hello': 4, 'world': 3, 'IO_TT': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            f = torchtext.vocab.FastText(language='simple')
            v = torchtext.legacy.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=f)
            n_vocab = len(v)
            v.extend(f)  # extend the vocab with the words contained in f.itos
            self.assertGreater(len(v), n_vocab)

            self.assertEqual(v.itos[:6], ['<unk>', '<pad>', '<bos>',
                                          'IO_TT', 'hello', 'world'])
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))

    @unittest.skip("Download temp. slow.")
    def test_vectors_custom_cache(self):
        c = Counter({'hello': 4, 'world': 3, 'IO_TT': 5, 'freq_too_low': 2})
        vector_cache = os.path.join('/tmp', 'vector_cache')
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            if i == 1:
                self.assertTrue(os.path.exists(vector_cache))

            v = torchtext.legacy.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                vectors=torchtext.vocab.Vectors(
                    'wiki.simple.vec', cache=vector_cache,
                    url=torchtext.vocab.FastText.url_base.format('simple'))
            )

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'IO_TT', 'hello', 'world'])
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))


"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""
import re

from torch.nn import Module

from ..model import Wav2Vec2Model, wav2vec2_model


def _parse_config(w2v_model):
    encoder = w2v_model.encoder
    conv_layers = w2v_model.feature_extractor.conv_layers

    extractor_mode = 'layer_norm'
    if 'GroupNorm' in conv_layers[0][2].__class__.__name__:
        extractor_mode = 'group_norm'
    else:
        extractor_mode = 'layer_norm'

    conv_layer_config = [(l[0].out_channels, l[0].kernel_size[0], l[0].stride[0]) for l in conv_layers]

    if all(l[0].bias is None for l in conv_layers):
        conv_bias = False
    elif all(l[0].bias is not None for l in conv_layers):
        conv_bias = True
    else:
        raise ValueError(
            'Either all the convolutions layers have bias term or none of them should.')

    config = {
        'extractor_mode': extractor_mode,
        'extractor_conv_layer_config': conv_layer_config,
        'extractor_conv_bias': conv_bias,
        'encoder_embed_dim': w2v_model.post_extract_proj.out_features,
        'encoder_projection_dropout': w2v_model.dropout_input.p,
        'encoder_pos_conv_kernel': encoder.pos_conv[0].kernel_size[0],
        'encoder_pos_conv_groups': encoder.pos_conv[0].groups,
        'encoder_num_layers': len(encoder.layers),
        'encoder_num_heads': encoder.layers[0].self_attn.num_heads,
        'encoder_attention_dropout': encoder.layers[0].self_attn.dropout_module.p,
        'encoder_ff_interm_features': encoder.layers[0].fc1.out_features,
        'encoder_ff_interm_dropout': encoder.layers[0].dropout2.p,
        'encoder_dropout': encoder.layers[0].dropout3.p,
        'encoder_layer_norm_first': encoder.layer_norm_first,
        'encoder_layer_drop': encoder.layerdrop,
    }
    return config


def _map_key(key):
    key_ = key
    if key.startswith('w2v_model.'):
        key = key.replace('w2v_model.', '')
    if re.match(r'(mask_emb|quantizer|project_q|final_proj|mask_emb)', key):
        return None
    # Feature Extractor
    # Group norm when "extractor_mode" is "default".
    # (Only the first layer)
    # "conv_layers.0.2.weight" -> "conv_layers.0.layer_norm.weight"
    # "conv_layers.0.2.bias"   -> "conv_layers.0.layer_norm.bias"
    match = re.match(r'feature_extractor\.conv_layers\.0\.2\.(weight|bias)', key)
    if match:
        return f"feature_extractor.conv_layers.0.layer_norm.{match.group(1)}"
    # Convolutions
    # "conv_layers.X.0.weight" -> "conv_layers.X.conv.weight"
    # "conv_layers.X.0.bias"   -> "conv_layers.X.conv.bias"
    match = re.match(r'feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)', key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.conv.{match.group(2)}"
    # Layer norm when "extractor_mode" is "layer_norm".
    # "conv_layers.X.2.1.weight" -> "conv_layers.X.layer_norm.weight"
    # "conv_layers.X.2.1.bias"   -> "conv_layers.X.layer_norm.bias"
    match = re.match(r'feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)', key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
    match = re.match(r"post_extract_proj\.(weight|bias)", key)
    # Encoder - Feature projection
    if match:
        return f"encoder.feature_projection.projection.{match.group(1)}"
    match = re.match(r"layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.feature_projection.layer_norm.{match.group(1)}"
    # Encoder - Transformer - Convolutional positional embedding
    match = re.match(r"encoder\.pos_conv\.0\.(bias|weight_g|weight_v)", key)
    if match:
        return f"encoder.transformer.pos_conv_embed.conv.{match.group(1)}"
    match = re.match(r"encoder\.layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layer_norm.{match.group(1)}"
    # Encoder - Transformer - Self attention layers
    match = re.match(r"encoder\.layers\.(\d+)\.self_attn\.((k_|v_|q_|out_)proj\.(weight|bias))", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.attention.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.layer_norm.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.fc1\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.feed_forward.intermediate_dense.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.fc2\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.feed_forward.output_dense.{match.group(2)}"
    match = re.match(r"encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)", key)
    if match:
        return f"encoder.transformer.layers.{match.group(1)}.final_layer_norm.{match.group(2)}"
    match = re.match(r"proj\.(weight|bias)", key)
    # Auxiliary Module
    # Only relevant when loading fine-tuned models
    if match:
        return f"aux.{match.group(1)}"
    # HuBERT Extension
    if key in ['label_embs_concat']:
        return key
    raise ValueError(f'Unexpected key: {key_}')


def _convert_state_dict(state_dict):
    converted = {}
    for k, v in state_dict.items():
        k = _map_key(k)
        if k is not None:
            converted[k] = v
    return converted


def import_fairseq_model(original: Module) -> Wav2Vec2Model:
    # Overriding the signature so that the types are correct on Sphinx
    """import_fairseq_model(original: torch.nn.Module) -> torchaudio.models.Wav2Vec2Model

    Build Wav2Vec2Model from the corresponding model object of `fairseq`_.

    Args:
        original (torch.nn.Module):
            An instance of fairseq's Wav2Vec2.0 or HuBERT model.
            One of ``fairseq.models.wav2vec.wav2vec2_asr.Wav2VecEncoder``,
            ``fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model`` or
            ``fairseq.models.hubert.hubert_asr.HubertEncoder``.

    Returns:
        Wav2Vec2Model: Imported model.

    Example - Loading pretrain-only model
        >>> from torchaudio.models.wav2vec2.utils import import_fairseq_model
        >>>
        >>> # Load model using fairseq
        >>> model_file = 'wav2vec_small.pt'
        >>> model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
        >>> original = model[0]
        >>> imported = import_fairseq_model(original)
        >>>
        >>> # Perform feature extraction
        >>> waveform, _ = torchaudio.load('audio.wav')
        >>> features, _ = imported.extract_features(waveform)
        >>>
        >>> # Compare result with the original model from fairseq
        >>> reference = original.feature_extractor(waveform).transpose(1, 2)
        >>> torch.testing.assert_allclose(features, reference)

    Example - Fine-tuned model
        >>> from torchaudio.models.wav2vec2.utils import import_fairseq_model
        >>>
        >>> # Load model using fairseq
        >>> model_file = 'wav2vec_small_960h.pt'
        >>> model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
        >>> original = model[0]
        >>> imported = import_fairseq_model(original.w2v_encoder)
        >>>
        >>> # Perform encoding
        >>> waveform, _ = torchaudio.load('audio.wav')
        >>> emission, _ = imported(waveform)
        >>>
        >>> # Compare result with the original model from fairseq
        >>> mask = torch.zeros_like(waveform)
        >>> reference = original(waveform, mask)['encoder_out'].transpose(0, 1)
        >>> torch.testing.assert_allclose(emission, reference)

    .. _fairseq: https://github.com/pytorch/fairseq
    """
    class_ = original.__class__.__name__
    if class_ == 'Wav2Vec2Model':
        return _import_wav2vec2_pretraining(original)
    if class_ == 'Wav2VecEncoder':
        return _import_wav2vec2_finetuning(original)
    if class_ == 'HubertModel':
        return _import_hubert_pretraining(original)
    if class_ == 'HubertEncoder':
        return _import_hubert_finetuning(original)
    raise ValueError(
        f'Expected an instance of `Wav2Vec2Model` or `Wav2VecEncoder`. Found: {class_}')


def _import_wav2vec2_finetuning(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original.w2v_model)
    model = wav2vec2_model(**config, aux_num_out=original.proj.out_features)
    model.load_state_dict(_convert_state_dict(original.state_dict()))
    return model


def _import_wav2vec2_pretraining(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original)
    model = wav2vec2_model(**config, aux_num_out=None)
    model.load_state_dict(_convert_state_dict(original.state_dict()), strict=False)
    return model


def _import_hubert_finetuning(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original.w2v_model)
    model = wav2vec2_model(**config, aux_num_out=original.proj.out_features)
    model.load_state_dict(_convert_state_dict(original.state_dict()), strict=False)
    return model


def _import_hubert_pretraining(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original)
    model = wav2vec2_model(**config, aux_num_out=None)
    model.load_state_dict(_convert_state_dict(original.state_dict()), strict=False)
    return model


#!/usr/bin/env python3
"""Convert a Wav2Vec2/HuBERT model published by fairseq into torchaudio format

Examples

```
python convert_fairseq_models.py \
  --input-file hubert_base_ls960.pt \
  --output-file hubert_fairseq_base_ls960.pth

python convert_fairseq_models.py \
  --input-file hubert_large_ll60k.pt \
  --output-file hubert_fairseq_large_ll60k.pth

python convert_fairseq_models.py \
  --input-file hubert_large_ll60k_finetune_ls960.pt \
  --output-file hubert_fairseq_large_ll60k_asr_ls960.pth

python convert_fairseq_models.py \
  --input-file hubert_xtralarge_ll60k.pt \
  --output-file hubert_fairseq_xlarge_ll60k.pth

python convert_fairseq_models.py \
  --input-file hubert_xtralarge_ll60k_finetune_ls960.pt \
  --output-file hubert_fairseq_xlarge_ll60k_asr_ls960.pth
"""

import argparse

# Note: Avoiding the import of torch and fairseq on global scope as they are slow


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--input-file', required=True,
        help='Input model file.'
    )
    parser.add_argument(
        '--output-file', required=False,
        help='Output model file.'
    )
    parser.add_argument(
        '--dict-dir',
        help=(
            'Directory where letter vocabulary file, `dict.ltr.txt`, is found. '
            'Required when loading wav2vec2 model. '
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt'
        )
    )
    return parser.parse_args()


def _load_model(input_file, dict_dir):
    import fairseq

    overrides = {} if dict_dir is None else {'data': dict_dir}
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [input_file], arg_overrides=overrides,
    )
    return models[0]


def _import_model(model):
    from torchaudio.models.wav2vec2.utils import import_fairseq_model

    if model.__class__.__name__ in ['HubertCtc', 'Wav2VecCtc']:
        model = model.w2v_encoder
    model = import_fairseq_model(model)
    return model


def _main(args):
    import torch
    model = _load_model(args.input_file, args.dict_dir)
    model = _import_model(model)
    torch.save(model.state_dict(), args.output_file)


if __name__ == '__main__':
    _main(_parse_args())


import torch
import torchtext.legacy.data as data

from ...common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    def test_batch_with_missing_field(self):
        # smoke test to see if batches with missing attributes are shown properly
        with open(self.test_missing_field_dataset_path, "wt") as f:
            f.write("text,label\n1,0")

        dst = data.TabularDataset(path=self.test_missing_field_dataset_path,
                                  format="csv", skip_header=True,
                                  fields=[("text", data.Field(use_vocab=False,
                                                              sequential=False)),
                                          ("label", None)])
        itr = data.Iterator(dst, batch_size=64)
        str(next(itr.__iter__()))

    def test_batch_iter(self):
        self.write_test_numerical_features_dataset()
        FLOAT = data.Field(use_vocab=False, sequential=False,
                           dtype=torch.float)
        INT = data.Field(use_vocab=False, sequential=False, is_target=True)
        TEXT = data.Field(sequential=False)

        dst = data.TabularDataset(path=self.test_numerical_features_dataset_path,
                                  format="tsv", skip_header=False,
                                  fields=[("float", FLOAT),
                                          ("int", INT),
                                          ("text", TEXT)])
        TEXT.build_vocab(dst)
        itr = data.Iterator(dst, batch_size=2, device=-1, shuffle=False)
        fld_order = [k for k, v in dst.fields.items() if
                     v is not None and not v.is_target]
        batch = next(iter(itr))
        (x1, x2), y = batch
        x = (x1, x2)[fld_order.index("float")]
        self.assertEqual(y.data[0], 1)
        self.assertEqual(y.data[1], 12)
        self.assertAlmostEqual(x.data[0], 0.1, places=4)
        self.assertAlmostEqual(x.data[1], 0.5, places=4)


from torchtext.legacy import datasets

# en-valid
TRAIN_NUM = [0] + [900] * 16 + [904, 905, 900, 904]
VAL_NUM = [0] + [100] * 16 + [96, 95, 100, 96]
TEST_NUM = [0] + [1000] * 20

# Testcase 1 (joint training)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, joint=True)
assert len(train_iter.dataset) == sum(TRAIN_NUM)
assert len(val_iter.dataset) == VAL_NUM[1]
assert len(test_iter.dataset) == TEST_NUM[1]

# Testcase 2 (only supporting)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, only_supporting=True)
assert len(train_iter.dataset) == TRAIN_NUM[2]
assert len(val_iter.dataset) == VAL_NUM[2]
assert len(test_iter.dataset) == TEST_NUM[2]

# Testcase 3 (single task)
for i in range(1, 21):
    train_iter, val_iter, test_iter = datasets.BABI20.iters(task=i)
    assert len(train_iter.dataset) == TRAIN_NUM[i]
    assert len(val_iter.dataset) == VAL_NUM[i]
    assert len(test_iter.dataset) == TEST_NUM[i]

# en-valid-10k
TRAIN_NUM = [0] + [9000] * 17 + [8996, 9000, 9002]
VAL_NUM = [0] + [1000] * 17 + [1004, 1000, 998]
TEST_NUM = [0] + [1000] * 20

# Testcase 1 (joint training)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, joint=True, tenK=True)
assert len(train_iter.dataset) == sum(TRAIN_NUM)
assert len(val_iter.dataset) == VAL_NUM[1]
assert len(test_iter.dataset) == TEST_NUM[1]

# Testcase 2 (only supporting)
train_iter, val_iter, test_iter = datasets.BABI20.iters(task=1, only_supporting=True,
                                                        tenK=True)
assert len(train_iter.dataset) == TRAIN_NUM[2]
assert len(val_iter.dataset) == VAL_NUM[2]
assert len(test_iter.dataset) == TEST_NUM[2]

# Testcase 3 (single task)
for i in range(1, 21):
    train_iter, val_iter, test_iter = datasets.BABI20.iters(task=i, tenK=True)
    assert len(train_iter.dataset) == TRAIN_NUM[i]
    assert len(val_iter.dataset) == VAL_NUM[i]
    assert len(test_iter.dataset) == TEST_NUM[i]




from pathlib import Path

import pytest

from torchaudio.datasets import dr_vctk

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
)


_SUBSETS = ["train", "test"]
_CONDITIONS = ["clean", "device-recorded"]
_SOURCES = ["DR-VCTK_Office1_ClosedWindow", "DR-VCTK_Office1_OpenedWindow"]
_SPEAKER_IDS = range(226, 230)
_CHANNEL_IDS = range(1, 6)


def get_mock_dataset(root_dir):
    """
    root_dir: root directory of the mocked data
    """
    mocked_samples = {}

    dataset_dir = Path(root_dir) / "DR-VCTK" / "DR-VCTK"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_dir = dataset_dir / "configurations"
    config_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    seed = 0

    for subset in _SUBSETS:
        mocked_samples[subset] = []

        for condition in _CONDITIONS:
            audio_dir = dataset_dir / f"{condition}_{subset}set_wav_16k"
            audio_dir.mkdir(parents=True, exist_ok=True)

        config_filepath = config_dir / f"{subset}_ch_log.txt"
        with open(config_filepath, "w") as f:
            if subset == "train":
                f.write("\n")
            f.write("File Name\tMain Source\tChannel Idx\n")

            for speaker_id in _SPEAKER_IDS:
                utterance_id = 1
                for source in _SOURCES:
                    for channel_id in _CHANNEL_IDS:
                        filename = f"p{speaker_id}_{utterance_id:03d}.wav"
                        f.write(f"{filename}\t{source}\t{channel_id}\n")

                        data = {}
                        for condition in _CONDITIONS:
                            data[condition] = get_whitenoise(
                                sample_rate=sample_rate,
                                duration=0.01,
                                n_channels=1,
                                dtype='float32',
                                seed=seed
                            )
                            audio_dir = dataset_dir / f"{condition}_{subset}set_wav_16k"
                            audio_file_path = audio_dir / filename
                            save_wav(audio_file_path, data[condition], sample_rate)
                            seed += 1

                        sample = (
                            data[_CONDITIONS[0]],
                            sample_rate,
                            data[_CONDITIONS[1]],
                            sample_rate,
                            "p" + str(speaker_id),
                            f"{utterance_id:03d}",
                            source,
                            channel_id,
                        )
                        mocked_samples[subset].append(sample)
                        utterance_id += 1

    return mocked_samples


class TestDRVCTK(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = get_mock_dataset(cls.root_dir)

    def _test_dr_vctk(self, dataset, subset):
        num_samples = 0
        for i, (
            waveform_clean,
            sample_rate_clean,
            waveform_dr,
            sample_rate_dr,
            speaker_id,
            utterance_id,
            source,
            channel_id,
        ) in enumerate(dataset):
            self.assertEqual(waveform_clean, self.samples[subset][i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate_clean == self.samples[subset][i][1]
            self.assertEqual(waveform_dr, self.samples[subset][i][2], atol=5e-5, rtol=1e-8)
            assert sample_rate_dr == self.samples[subset][i][3]
            assert speaker_id == self.samples[subset][i][4]
            assert utterance_id == self.samples[subset][i][5]
            assert source == self.samples[subset][i][6]
            assert channel_id == self.samples[subset][i][7]

            num_samples += 1

        assert num_samples == len(self.samples[subset])

    def test_dr_vctk_train_str(self):
        subset = "train"
        dataset = dr_vctk.DR_VCTK(self.root_dir, subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_test_str(self):
        subset = "test"
        dataset = dr_vctk.DR_VCTK(self.root_dir, subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_train_path(self):
        subset = "train"
        dataset = dr_vctk.DR_VCTK(Path(self.root_dir), subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_test_path(self):
        subset = "test"
        dataset = dr_vctk.DR_VCTK(Path(self.root_dir), subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_invalid_subset(self):
        subset = "invalid"
        with pytest.raises(RuntimeError, match=f"The subset '{subset}' does not match any of the supported subsets"):
            dr_vctk.DR_VCTK(self.root_dir, subset=subset)


import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

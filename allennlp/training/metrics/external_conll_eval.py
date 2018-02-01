from typing import Dict, List, Optional, Set, Tuple, TextIO
from collections import defaultdict

import torch
import subprocess

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, ones_like
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric

import IPython as ipy

@Metric.register("conll_f1")
class ExternalConllEval(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model peformance during training.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "labels",
                 psd_namespace: str = "senses",
                 ignore_classes: List[str] = None) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        """
        self.vocabulary = vocabulary
        self._tag_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._sense_vocabulary = vocabulary.get_index_to_token_vocabulary(psd_namespace)
        self._token_vocabulary = self.vocabulary.get_index_to_token_vocabulary('tokens')
        self.oov = vocabulary._oov_token
        self._ignore_classes = ignore_classes or []

        self.predict_filename = 'metric_predictions.conll'
        self.gold_filename = 'metric_gold.conll'

        self.all_words = {}
        self.all_predicate_inds = defaultdict(list)
        self.all_gold_senses = defaultdict(list)
        self.all_predicted_senses = defaultdict(list)
        self.all_gold_tags = defaultdict(list)
        self.all_predicted_tags = defaultdict(list)


    def __call__(self, decode_output: Dict):
        """
        Parameters
        ----------
        decode_output: the output dictionary from model.decode()
        """
        tokens = decode_output["tokens"]["tokens"].data
        batch_len = len(tokens)
        
        for i in range(batch_len):
            words = []
            for v in tokens[i]:
                if v != 0:
                    word = self._token_vocabulary[v]
                    words.append(word)
            length = len(words)

            pred_indices = list(decode_output['pred_indicator'].data[i])[:length]

            gold_tags = []
            for v in decode_output['gold_tags'].data[i][:length]:
                tag = 'O'
                if v != 0: tag = self._tag_vocabulary[v]
                gold_tags.append(tag)

            gold_senses = []
            for v in decode_output['gold_sense'].data[i]:
                sense = self._sense_vocabulary[v]
                gold_senses.append(sense)

            predicted_tags = decode_output['tags'][i]
            predicted_sense = decode_output['sense'][i]
            if predicted_sense == self.oov:
                # not a real predicate sense, because we didn't recognize the predicate
                # guess it with a heuristic
                tok_lemma = decode_output['pred_sense_set']
                print("\n Working on predicate heuristic \n")
                ipy.embed()
                predicted_sense = tok_lemma.split(':')[-1] + '.01'

            sid = decode_output['metadata'][i]['sentence_id']
            if sid in self.all_words:
                try:
                    assert self.all_words[sid] == words
                except:
                    print("\n Word sequence mismatch between instances with the same sentence ID\n")
                    ipy.embed()
            else:
                self.all_words[sid] = words

            self.all_predicate_inds[sid].append(pred_indices)
            self.all_gold_senses[sid] += gold_senses
            self.all_predicted_senses[sid].append(predicted_sense)
            self.all_gold_tags[sid].append(gold_tags)
            self.all_predicted_tags[sid].append(predicted_tags)


    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        predict_file = open(self.predict_filename, 'w')
        gold_file = open(self.gold_filename, 'w')
        for sid in self.all_words:
            write_to_conll_2009_eval_file(predict_file, gold_file,
                                          self.all_words[sid],
                                          self.all_predicate_inds[sid],
                                          self.all_gold_senses[sid],
                                          self.all_predicted_senses[sid],
                                          self.all_gold_tags[sid],
                                          self.all_predicted_tags[sid])
        predict_file.close()
        gold_file.close()

        perl_eval = subprocess.run(['perl', 'scripts/eval09.pl', 
                                    '-g', self.gold_filename, 
                                    '-s', self.predict_filename],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        output = perl_eval.stdout.decode('utf-8').split('\n')
        precision, recall, f1_measure = self.parse_eval09(output)

        if reset:
            self.reset()

        all_metrics = {}
        all_metrics["precision-overall"] = float(precision)
        all_metrics["recall-overall"] = float(recall)
        all_metrics["f1-measure-overall"] = float(f1_measure)
        return all_metrics


    @staticmethod
    def parse_eval09(lines):
        precision, recall, f1 = None, None, None
        for line in lines:
            if "Labeled precision:" in line:
                precision = line.split()[-2]
            if "Labeled recall:" in line:
                recall = line.split()[-2]
            if "Labeled F1:" in line:
                f1 = line.split()[-1]
            if precision and recall and f1:
                break
        return precision, recall, f1


    def reset(self):
        self.all_words = {}
        self.all_predicate_inds = defaultdict(list)
        self.all_gold_senses = defaultdict(list)
        self.all_predicted_senses = defaultdict(list)
        self.all_gold_tags = defaultdict(list)
        self.all_predicted_tags = defaultdict(list)


def write_to_conll_2009_eval_file(prediction_file: TextIO,
                                  gold_file: TextIO,
                                  sentence: List[str],
                                  pred_indices: List[List[int]],
                                  gold_senses: List[List[str]],
                                  predicted_senses: List[List[str]],
                                  gold_tags: List[List[str]],
                                  predicted_tags: List[List[str]],
                                  pos_tags: Optional[List[str]] = None):
    """
    Prints predicate argument predictions and optionally gold labels for a single 
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    gold_senses : List[List[str]]
        The gold predicate senses.
    pred_indices : Optional[int], required.
        The index of the predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no predicate.
    sentence : List[str], required.
        The word tokens.
    predicted_tags : List[str], required.
        The predicted BIO labels.
    gold_tags : List[str], required.
        The gold BIO labels.
    """
    pred_only_sentence = ["_"] * len(sentence)
    gold_only_sentence = ["_"] * len(sentence)
    pred_indicators = ["_"] * len(sentence)

    for i, pred_index_set in enumerate(pred_indices):
        for pidx, val in enumerate(pred_index_set):
            if val:
                pred_only_sentence[pidx] = predicted_senses[i]
                if len(gold_senses) > 0:
                    gold_only_sentence[pidx] = gold_senses[i]
                pred_indicators[pidx] = 'Y'

    lines = []
    for idx in range(len(sentence)):
        word = sentence[idx]
        line = ["_"] * (14 + len(pred_indices))
        line[0] = str(idx+1)
        line[1] = word

        if pos_tags is not None:
            line[4] = pos_tags[idx]
            line[5] = pos_tags[idx]

        if pred_indicators[idx] == 'Y':
            line[12] = pred_indicators[idx]
            line[13] = pred_only_sentence[idx]
        for i, predicate_tags in enumerate(predicted_tags):
            if predicate_tags[idx] != 'O':
                tag = predicate_tags[idx]
                line[14+i] = '-'.join(tag.split('-')[1:]) # remove the B- from the beginning of the tag
        prediction_file.write('\t'.join(line)+'\n')
        prediction_file.flush()
        lines.append(line)

        if pred_indicators[idx] == 'Y':
            line[13] = gold_only_sentence[idx]
        for i, predicate_tags in enumerate(gold_tags):
            tag = predicate_tags[idx]
            if 'B-' in tag[:2]: tag = '-'.join(tag.split('-')[1:])
            line[14+i] = tag
        gold_file.write('\t'.join(line)+'\n')
        gold_file.flush()

    prediction_file.write("\n")
    gold_file.write("\n")



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
                 filename: str = "metric",
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

        self.predict_filename = filename + "_predictions.conll"
        self.gold_filename = filename + "_gold.conll"

        self.all_words = {}
        self.all_predicate_inds = defaultdict(list)
        self.all_gold_senses = defaultdict(list)
        self.all_predicted_senses = defaultdict(list)
        self.all_gold_tags = defaultdict(list)
        self.all_predicted_tags = defaultdict(list)
        self.populated = False

    def __call__(self,
                 token_inds,
                 pred_indicators,
                 gold_tags,
                 tag_probabilities,
                 pred_sense_sets,
                 gold_senses,
                 sense_predictions,
                 sentence_ids):
        """
        Parameters
        ----------
        decode_output: the output dictionary from model.decode()
        """
        self.populated = True
        token_vals = token_inds["tokens"].data
        batch_len = len(token_vals)

        tag_predictions = tag_probabilities.data.max(-1)[1]
        
        for i in range(batch_len):

            sid = sentence_ids[i]['sentence_id']
            gold_sense = self._sense_vocabulary[gold_senses.data[i,0]]
            if gold_sense in self.all_gold_senses[sid]:
                # this instance is a duplicate of one we've seen before
                # keep the first prediction and discard later ones
                continue

            words = []
            for v in token_vals[i]:
                if v != 0:
                    word = self._token_vocabulary[v]
                    words.append(word)
            length = len(words)
            #if sid in self.all_words:
            #    assert self.all_words[sid] == words
            #else:
            self.all_words[sid] = words

            pred_indices = list(pred_indicators.data[i])[:length]

            gold_tag_list = []
            for v in gold_tags.data[i][:length]:
                tag = 'O'
                if v != 0: tag = self._tag_vocabulary[v]
                gold_tag_list.append(tag)

            tag_strings = [self._tag_vocabulary[t] for t in tag_predictions[i][:length]]

            self.all_predicate_inds[sid].append(pred_indices)
            self.all_gold_senses[sid].append(gold_sense)
            self.all_predicted_senses[sid].append(sense_predictions[i])
            self.all_gold_tags[sid].append(gold_tag_list)
            self.all_predicted_tags[sid].append(tag_strings)


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
        sorted_keys = sorted(self.all_words.keys())
        for sid in sorted_keys:
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
        if precision is None or recall is None or f1_measure is None:
            print("\nERROR: could not read eval file")
            return all_metrics

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
        self.populated = False

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

    predicates_by_idx = {}
    empty_pred_count = 0
    for i, pred_index_set in enumerate(pred_indices):
        if 1 in pred_index_set:
            predicates_by_idx[pred_index_set.index(1)] = i
        else:
            empty_pred_count += 1
        for pidx, val in enumerate(pred_index_set):
            if val:
                pred_only_sentence[pidx] = predicted_senses[i]
                if len(gold_senses) > 0:
                    gold_only_sentence[pidx] = gold_senses[i]
                pred_indicators[pidx] = 'Y'
    if empty_pred_count > 1 or empty_pred_count > 0 and len(predicates_by_idx) > 0:
        print("Error in printing CoNLL file: Multiple predicates and one or more is empty")
        ipy.embed()

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
        for predicate_num, predicate_idx in enumerate(sorted(predicates_by_idx.keys())):
            unsorted_num = predicates_by_idx[predicate_idx]
            predicate_tags = predicted_tags[unsorted_num]
            if predicate_tags[idx] != 'O':
                tag = predicate_tags[idx]
                if '~' in tag:
                    tag = '~'.join(tag.split('~')[:-1])
                line[14+predicate_num] = '-'.join(tag.split('-')[1:]) # remove the B- from the beginning of the tag
        try:
            prediction_file.write('\t'.join(line)+'\n')
        except:
            ipy.embed()
        prediction_file.flush()
        lines.append(line)

        if pred_indicators[idx] == 'Y':
            line[13] = gold_only_sentence[idx]

        for predicate_num, predicate_idx in enumerate(sorted(predicates_by_idx.keys())):
            unsorted_num = predicates_by_idx[predicate_idx]
            predicate_tags = gold_tags[unsorted_num]
            tag = predicate_tags[idx]
            if tag == 'O':
                tag = '_'
            elif 'B-' in tag[:2]:
                tag = '-'.join(tag.split('-')[1:])
            if '~' in tag:
                tag = '~'.join(tag.split('~')[:-1])
            line[14+predicate_num] = tag
        gold_file.write('\t'.join(line)+'\n')
        gold_file.flush()

    prediction_file.write("\n")
    gold_file.write("\n")



def write_predicates_to_conll_file(prediction_file: TextIO,
                                   gold_file: TextIO,
                                   sentence: List[str],
                                   pos_tags: List[str],
                                   pred_values: List[str],
                                   gold_pred_indicator: List[int]):
    """
    Prints predicate argument predictions and optionally gold labels for a single 
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    sentence : List[str], required.
        The word tokens.
    pos : List[str], required.
        The part-of-speech tags..
    pred_values : List[str], required.
        The values of the FILLPRED column: 'Y' if a predicate, '_' else.
    """
    lines = []
    npreds = len([p for p in pred_values if p=='Y'])

    for idx in range(len(sentence)):
        word = sentence[idx]
        line = ["_"] * (14 + npreds)
        line[0] = str(idx+1)
        line[1] = word
        line[2] = word
        line[3] = word

        postag = pos_tags[idx]
        line[4] = postag
        line[5] = postag

        if ':' in word and len(word.split(':')[0]) == 3:
            word = ':'.join(word.split(':')[1:])

        if pred_values[idx] == 'Y':
            line[12] = pred_values[idx]
            line[13] = word

        prediction_file.write("\t".join(line)+"\n")
        prediction_file.flush()
        lines.append(line)        

    # ignore gold bc there's no way to compare sentences with 
    # varying numbers of predicates anyway

    prediction_file.write("\n")
    #gold_file.write("\n")



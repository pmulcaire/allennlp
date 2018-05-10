from typing import Dict, List, TextIO, Optional, Any

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
from torch.nn.parameter import Parameter
from torch.autograd.variable import Variable
import torch.nn.functional as F
import math
import numpy as np

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, masked_cross_entropy
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import ExternalConllEval, SpanBasedF1Measure, F1Measure

import IPython as ipy

@Model.register("predid")
class PredicateIdentification(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implmentation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(PredicateIdentification, self).__init__(vocab, regularizer)

        self.dev_id = None

        self.text_field_embedder = text_field_embedder
        self.num_classes = 2

        # For the span based evaluation, we don't want to consider labels
        # for the predicate, because the predicate index is provided to the model.
        metric_string = "base_metric"
        #self.conll_metric = ExternalConllEval(vocab, filename=metric_string, ignore_classes=["V"])
        self.metric = F1Measure(positive_label=1)

        self.stacked_encoder = stacked_encoder
        self.binary_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)

        if text_field_embedder.get_output_dim()  != stacked_encoder.get_input_dim():
            raise ConfigurationError("The input dimension of the stacked_encoder must be equal to "
                                     "the output dimension of the text_field_embedder")

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pred_indicator: torch.LongTensor,
                metadata: List[Dict[str,Any]] = None,
                calculate_loss: bool = True) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pred_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the predicate
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no predicate.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens)) # (batch_size, sequence_length, embedding_size)
        mask = get_text_field_mask(tokens) # (batch_size, sequence_length)
        batch_size, sequence_length, embedding_dim = embedded_text_input.size()

        if self.stacked_encoder.get_input_dim() != embedding_dim:
            raise ConfigurationError("The 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim.")

        encoded_text = self.stacked_encoder(embedded_text_input, mask)

        logits = self.binary_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, 
                                                                  sequence_length, 
                                                                  self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if pred_indicator is not None and calculate_loss:
            arg_loss = sequence_cross_entropy_with_logits(logits, pred_indicator, mask)
            self.metric(class_probabilities, pred_indicator, mask)
            output_dict["loss"] = arg_loss

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        output_dict = self.decode_predicates(output_dict)
        return output_dict

    def decode_predicates(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions, length in zip(predictions_list, sequence_lengths):
            vals, inds = predictions[:length].max(1)
            pred_inds = ['_' if idx == 0 else 'Y' for idx in inds]
            all_tags.append(pred_inds)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        p,r,f1 = self.metric.get_metric(reset=reset)
        metric_dict = {"f1-measure-overall":f1,"precision":p,"recall":r}
        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'PredicateIdentification':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   initializer=initializer,
                   regularizer=regularizer)

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.dev_id = device_id
        return super(PredicateIdentification, self).cuda(device_id=self.dev_id)


def write_to_conll_2012_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             pred_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single 
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    pred_index : Optional[int], required.
        The index of the predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    pred_only_sentence = ["-"] * len(sentence)
    if pred_index:
        pred_only_sentence[pred_index] = sentence[pred_index]

    conll_format_predictions = convert_bio_tags_to_conll_2012_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_2012_format(gold_labels)

    for word, predicted, gold in zip(pred_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_2012_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels

def write_to_conll_2009_eval_file(prediction_file: TextIO,
                                  gold_file: TextIO,
                                  pred_indices: List[List[int]],
                                  gold_senses: List[List[str]],
                                  predicted_senses: List[List[str]],
                                  sentence: List[str],
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
        word = sentence[idx].text
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
            if predicate_tags[idx] != 'O':
                tag = predicate_tags[idx]
                line[14+i] = '-'.join(tag.split('-')[1:]) # remove the B- from the beginning of the tag
        gold_file.write('\t'.join(line)+'\n')
        gold_file.flush()

    prediction_file.write("\n")
    gold_file.write("\n")



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
from allennlp.training.metrics import ExternalConllEval, SpanBasedF1Measure

import IPython as ipy

@Model.register("srl09")
class SemanticRoleLabeler(Model):
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
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SemanticRoleLabeler, self).__init__(vocab, regularizer)

        self.dev_id = None

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.num_senses = self.vocab.get_vocab_size("senses")

        # For the span based evaluation, we don't want to consider labels
        # for the predicate, because the predicate index is provided to the model.
        self.conll_metric = ExternalConllEval(vocab, ignore_classes=["V"])
        self.span_metric = SpanBasedF1Measure(vocab, ignore_classes=["V"])

        self.stacked_encoder = stacked_encoder
        # There are exactly 2 binary features for the predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                           self.num_classes))
        self.sense_weights = Parameter(torch.Tensor(self.num_senses, 
                                                    self.stacked_encoder.get_output_dim()))
        self.sense_bias = self.bias = Parameter(torch.Tensor(self.num_senses))
        stdv = 1. / math.sqrt(self.sense_weights.size(1))
        self.sense_weights.data.uniform_(-stdv, stdv)
        self.sense_bias.data.uniform_(-stdv, stdv)

        self.embedding_dropout = Dropout(p=embedding_dropout)

        if text_field_embedder.get_output_dim() + binary_feature_dim != stacked_encoder.get_input_dim():
            raise ConfigurationError("The SRL Model uses a binary predicate indicator feature, meaning "
                                     "the input dimension of the stacked_encoder must be equal to "
                                     "the output dimension of the text_field_embedder + 1.")

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                pred_indicator: torch.LongTensor,
                pred_sense_set: torch.LongTensor,
                pred_sense: torch.LongTensor = None,
                tags: torch.LongTensor = None,
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
        pred_sense_set: torch.LongTensor, required.
            A torch tensor representing the indices for valid pred_sense predictions. 
        pred_sense: torch.LongTensor, optional (default = None)
            A torch tensor representing the integer gold sense labels of shape ``(batch_size, 1)``
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``

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
        embedded_pred_indicator = self.binary_feature_embedding(pred_indicator.long()) # (batch_size, sequence_length, embedding_size)
        # Concatenate the predicate feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_pred_indicator = torch.cat([embedded_text_input, embedded_pred_indicator], -1)
        batch_size, sequence_length, embedding_dim_with_binary_feature = embedded_text_with_pred_indicator.size()

        if self.stacked_encoder.get_input_dim() != embedding_dim_with_binary_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes "
                                     "the embedding dimension one larger than the value "
                                     "specified. Therefore, the 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim + 1.")

        encoded_text = self.stacked_encoder(embedded_text_with_pred_indicator, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, 
                                                                  sequence_length, 
                                                                  self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None and calculate_loss:
            arg_loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            output_dict["arg_loss"] = arg_loss

        """-------------------------------------------------------------------------------------------------"""

        """
        Get predicate sense predictions for the batch
        PSL: pred sense lemma
        PSD: pred sense disambiguation
        """

        # predicate sense encodings for valid senses
        _, psl_inds = pred_indicator.max(1) # 1-D, shape (batch_size), values index sequence_length dimension
        d0, d2 = encoded_text.size(0), encoded_text.size(2)
        psl_inds_expanded = psl_inds.view(-1, 1, 1).expand(d0, 1, d2) # expand into shape (batch_size,1,embedding_size) for indexing
        psl_encodings = encoded_text.gather(1, psl_inds_expanded) # (batch_size, embedding_size)

        # create mask
        psd_mask = pred_sense_set.data.gt(-1)

        # select valid predicate sense embeddings
        compact_size = pred_sense_set.data.size(1)
        embedding_size = self.sense_weights.size(1)
        valid_sets = []
        valid_inds = []
        for rowi in range(batch_size):
            row_inds = Variable(pred_sense_set.data[rowi][psd_mask[rowi]].contiguous())
            valid_set = self.sense_weights.index_select(0,row_inds)
            if len(row_inds) < compact_size:
                zeros = np.zeros((compact_size - row_inds.size(0), embedding_size))
                if self.dev_id is not None:
                    emb_padding = Variable(torch.cuda.FloatTensor(zeros,device=self.dev_id))
                else:
                    emb_padding = Variable(torch.FloatTensor(zeros))
                valid_set = torch.cat([valid_set, emb_padding],0)
                ind_negs = -np.ones((compact_size - row_inds.size(0),),np.int64)
                if self.dev_id is not None:
                    ind_padding = Variable(torch.from_numpy(ind_negs).cuda(device=self.dev_id))
                else:
                    ind_padding = Variable(torch.from_numpy(ind_negs))
                row_inds = torch.cat([row_inds, ind_padding],0)
            valid_sets.append(valid_set)
            valid_inds.append(row_inds)
        sense_embs = torch.stack(valid_sets)
        scatter_inds = torch.stack(valid_inds)

        # project forward into logits
        # (batch_size x compact_size)
        psd_logits = torch.matmul(sense_embs, psl_encodings.permute(0,2,1))
        # set padding values to negative infinity
        if self.dev_id is not None:
            negative_mask = Variable(torch.cuda.FloatTensor(psd_logits.size(), device=self.dev_id).zero_())
        else:
            negative_mask = Variable(torch.FloatTensor(psd_logits.size()).zero_())
        negative_mask[psd_mask.lt(1)] = -np.inf
        masked_psd_logits = psd_logits + negative_mask
        compact_psd_probabilities = F.softmax(masked_psd_logits.view(batch_size, compact_size))
        psd_probabilities = compact_psd_probabilities #TODO FIX. project to full sense dimension, probably?
        psd_logits.squeeze()
        output_dict["psd_logits"] = psd_logits
        output_dict["psd_probabilities"] = psd_probabilities
        output_dict["psd_scatter_inds"] = scatter_inds

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        if pred_sense is not None and calculate_loss: #i.e. we have a gold answer and can backprop the loss from our predictions
            _, targets = torch.eq(scatter_inds,pred_sense).max(1)
            psd_loss = F.cross_entropy(psd_logits.view(batch_size, compact_size), targets, size_average=True)
            output_dict["psd_loss"] = psd_loss
            # combine loss, for backprop interface
            loss = arg_loss + psd_loss
            output_dict["loss"] = loss
            decode_output = self.decode(output_dict)
            if not self.training:
                self.conll_metric(token_inds=tokens,
                                  pred_indicators=pred_indicator,
                                  tags=tags,
                                  pred_sense_sets=pred_sense_set,
                                  senses=pred_sense,
                                  sentence_ids=metadata,
                                  decode_output=decode_output)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags

        sense_probabilities = output_dict['psd_probabilities']
        max_prob, indices = torch.max(sense_probabilities,1)
        all_senses = []
        for row, index in enumerate(indices.data):
            prediction_index = output_dict['psd_scatter_inds'][row,index].data[0]
            predicted_sense = self.vocab.get_token_from_index(prediction_index, namespace="senses")
            all_senses.append(predicted_sense)
        output_dict['sense'] = all_senses

        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = {}
        if self.conll_metric.populated:
            metric_dict = self.conll_metric.get_metric(reset=reset)
        metric_dict2 = self.span_metric.get_metric(reset=reset)
        if "f1-measure-overall" in metric_dict2:
            metric_dict['span-f1-overall'] = metric_dict2["f1-measure-overall"]
        if self.training:
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
            # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
            return {x: y for x, y in metric_dict.items() if "overall" in x}
        else:
            # limit eval output too
            return {x: y for x, y in metric_dict.items() if "overall" in x}

        return metric_dict

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The main constraint implemented is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. To achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Secondary constraints for predicate sense disambiguation: 
        1) the predicate index must have a tag corresponding to one of the 
           senses of the predicate.
        2) other words should not get a predicate sense tag, only BIO arg tags.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SemanticRoleLabeler':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        binary_feature_dim = params.pop("binary_feature_dim")

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   binary_feature_dim=binary_feature_dim,
                   initializer=initializer,
                   regularizer=regularizer)

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.dev_id = device_id
        return super(SemanticRoleLabeler, self).cuda()


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



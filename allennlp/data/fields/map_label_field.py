from typing import Dict, Union, List
import logging

from overrides import overrides
import numpy

from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields.label_field import LabelField
from allennlp.data.fields.list_field import ListField
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import IPython as ipy

class MapLabelField(ListField):
    """
    A ``MapLabelField`` extends the ``LabelField`` class (for categorical labels which are either 
    strings of text or 0-indexed integers) by allowing a single field to be indexed in multiple
    namespaces. This allows the representation of a mapping between different sets of labels (for
    example, to identify a set of possible options for a prediction to choose among).

    Like a ListField, this field will get converted into a tensor that has one more mode than the 
    items in the list (from target_namespace).
    If this is a list of ``TextFields`` that have shape (num_words, num_characters), this
    ``ListField`` will output a tensor of shape (num_sentences, num_words, num_characters).

    Parameters
    ----------
    index_label : ``Union[str, int]``
    target_label : ``Union[str, int]``
    index_label_namespace : ``str``, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    target_label_namespace : ``str``, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : ``bool``, optional (default=False)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is ``False`` and your labels are not strings, this throws a ``ConfigurationError``.
        For a MapLabelField this option is not recommended.
    """
    def __init__(self,
                 index_label: Union[str, int],
                 set_label: Union[str, int],
                 target_label: Union[str, int],
                 index_label_namespace: str = 'primary_labels',
                 set_label_namespace: str = 'secondary_labels',
                 target_label_namespace: str = 'tertiary_labels',
                 set_index_map: str = None,
                 set_target_map: str = None,
                 skip_indexing: bool = False) -> None:
        self.index_label = index_label # predicate token
        self.set_label = set_label # predicate lemma
        self.target_label = target_label # predicate sense

        self.target_label_set = set()

        self._index_label_namespace = index_label_namespace
        self._set_label_namespace = set_label_namespace
        self._target_label_namespace = target_label_namespace

        if set_index_map is None:
            self._set_index_map = index_label_namespace + '_map'
        if set_target_map is None:
            self._set_target_map = set_label_namespace + '_map'

        self._index_label_id = None
        self._set_label_id = None
        self._target_label_id = None

        namespace_err = None
        if not (self._set_label_namespace.endswith("labels") or self._set_label_namespace.endswith("tags")):
            namespace_err = 'primary'
            namespace = self._set_label_namespace
        if not (self._target_label_namespace.endswith("labels") or self._target_label_namespace.endswith("tags")):
            namespace_err = 'secondary'
            namespace = self._target_label_namespace
        if namespace_err is not None:
            logger.warning("Your {} label namespace was '{}'. ".format(namespace_err, namespace) +
                           "We recommend you use a namespace ending with 'labels' or 'tags', " +
                           "so we don't add UNK and PAD tokens by default to your vocabulary.  " +
                           "See documentation for `non_padded_namespaces` parameter in Vocabulary.",
                           self._set_label_namespace)

        if skip_indexing:
            if not isinstance(index_label, int) and isinstance(target_label, int):
                raise ConfigurationError("In order to skip indexing, your labels must be integers. "
                                         "Found index_label = {}, target_label = {}".format(index_label, target_label))
            else:
                self._index_label_id = index_label
                self.target_label_set = set([target_label])
        else:
            if not isinstance(index_label, str):
                raise ConfigurationError("LabelFields must be passed a string label if skip_indexing=False. "
                                         "Found label: {} with type: {}.".format(index_label, type(index_label)))

    @overrides
    def count_vocab_items(self, indexer: Dict[str, Dict[str, Union[int, List[int]]]]):
        # target_label is counted elsewhere
        if self._index_label_id is None:
            # count set_label normally here
            indexer[self._index_label_namespace][self.index_label] += 1  # type: ignore
        if self._set_label_id is None:
            # count set_label normally here
            indexer[self._set_label_namespace][self.set_label] += 1  # type: ignore
        if len(self.target_label_set) == 0:
            # ''count'' target and index labels here by adding set_label to the maps
            try:
                if self.set_label not in indexer[self._set_index_map]:
                    #override the 'int' default value of the defaultdict
                    indexer[self._set_index_map][self.set_label] = []
                if self.index_label not in indexer[self._set_index_map][self.set_label]:
                    indexer[self._set_index_map][self.set_label].append(self.index_label) # type: ignore
                if self.set_label not in indexer[self._set_target_map]:
                    #override the 'int' default value of the defaultdict
                    indexer[self._set_target_map][self.set_label] = []
                if self.target_label not in indexer[self._set_target_map][self.set_label]:
                    indexer[self._set_target_map][self.set_label].append(self.target_label) # type: ignore
            except:
                print("Problem with constructing maps in MapLabelField")
                ipy.embed()
        

    @overrides
    def index(self, vocab: Vocabulary):
        if self._index_label_id is None:
            self._index_label_id = vocab.get_token_index(self.index_label, self._index_label_namespace)  # type: ignore
        if self._set_label_id is None:
            self._set_label_id = vocab.get_token_index(self.set_label, self._set_label_namespace)  # type: ignore
        if len(self.target_label_set) == 0:
            set_set = vocab.get_token_from_index(self.index_label, self._set_index_map)
            flat_set = []
            for tgt_set in set_set:
                flat_set += vocab.get_token_index(tgt_set, self._set_target_map)  # type: ignore
            self.target_label_set = set(flat_set)
            target_ind_set = [vocab.get_token_index(lb, self._target_label_namespace) for lb in self.target_label_set]
            self.target_ind_set = sorted(list(target_ind_set))

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # sequences are currently not supported; see ListField for a template to 
        # implement sequence padding
        padding_lengths = {'num_fields': len(self.target_label_set)}
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self.target_label_set)

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:
        padded_label_list = pad_sequence_to_length(self.target_ind_set,
                                                   padding_lengths['num_fields'],
                                                   lambda: -1)
        return numpy.expand_dims(numpy.asarray(padded_label_list),1)

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        raise RuntimeError("Nested MapLabelFields are not implemented, and if you want this "
                           "you should /definitely/ try to simplify your data type, anyway")

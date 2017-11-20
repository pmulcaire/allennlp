import codecs
import os
import logging
from typing import Dict, List, Optional

from overrides import overrides
import tqdm
from collections import defaultdict

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

import IPython as ipy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("srl09")
class SrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in data in the format used by the 
    CoNLL 2008/2009 SRL shared tasks. You should structure your directories as
    follows:

    conll09-formatted-data/
       ├── development
           └── english
               ├── file1
               ├── ... [other files]
               └── fileN
           ├── ... [other languages]
           └── chinese
               ├── file1
               ├── ... [other files]
               └── fileN
       ├── test
           └── english
               ├── file1
               ├── ... [other files]
               └── fileN
           ├── ... [other languages]
           └── chinese
               ├── file1
               ├── ... [other files]
               └── fileN
       └── train
           └── english
               ├── file1
               ├── ... [other files]
               └── fileN
           ├── ... [other languages]
           └── chinese
               ├── file1
               ├── ... [other files]
               └── fileN

    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).

    The data has the following format, ordered by column.

    1 Document ID : str
        This is a variation on the document filename
    2 Part number : int
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3 Word number : int
        This is the word index of the word in that sentence.
    4 Word : str
        This is the token as segmented/tokenized in the Treebank. Initially the ``*_skel`` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    5 POS Tag : str
        This is the Penn Treebank style part of speech. When parse information is missing,
        all part of speeches except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    6 Parse bit: str
        This is the bracketed structure broken before the first open parenthesis in the parse,
        and the word/part-of-speech leaf replaced with a ``*``. The full parse can be created by
        substituting the asterisk with the "([pos] [word])" string (or leaf) and concatenating
        the items in the rows of that column. When the parse information is missing, the
        first word of a sentence is tagged as ``(TOP*`` and the last word is tagged as ``*)``
        and all intermediate words are tagged with a ``*``.
    7 Predicate lemma: str
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    8 Predicate Frameset ID: int
        The PropBank frameset ID of the predicate in Column 7.
    9 Word sense: float
        This is the word sense of the word in Column 3.
    10 Speaker/Author: str
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    11 Named Entities: str
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an ``*``.
    12+ Predicate Arguments: str
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    -1 Co-reference: str
        Co-reference chain information encoded in a parenthesis structure. For documents that do
         not have co-reference annotations, each line is represented with a "-".

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          predicate_indices: List[int],
                          predicate_senses: List[int],
                          predicate_argument_labels: List[List[str]],
                          sentence_id: int) -> List[Instance]:
        """
        Parameters
        ----------
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        predicates : ``List[int]``, required.
            The indexes of the predicates (both verbal and nominal) in the
            sentence which have an associated annotation.
        predicate_argument_labels : ``List[List[str]]``, required.
            A list of lists of predicate argument labels, one for each predicate. The
            internal lists are of length: len(sentence).

        Returns
        -------
        A list of Instances.

        """
        tokens = [Token(t) for t in sentence_tokens]
        if not predicate_indices:
            # Sentence contains no predicates.
            tags = ["O" for _ in sentence_tokens]
            pred_label = [0 for _ in sentence_tokens]
            pred_sense = "_"
            instances = [self.text_to_instance(tokens, pred_label, pred_sense, tags, sentence_id)]
        else:
            instances = []
            for pred_index, pred_sense, annotation in zip(predicate_indices,
                                                          predicate_senses,
                                                          predicate_argument_labels):
                tags = annotation
                pred_label = [0 for _ in sentence_tokens]
                pred_label[pred_index] = 1
                instance = self.text_to_instance(tokens, pred_label, pred_sense, tags, sentence_id)
                instances.append(instance)
        instance_count = len(instances)
        for instance in instances:
            instance.instance_count = instance_count
        return instances

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        sentence: List[str] = []
        predicates: List[int] = []
        senses: List[str] = []
        predicate_argument_labels: List[List[str]] = []
        sentence_count = 0
        token_to_senses = defaultdict(set)

        logger.info("Reading SRL instances from dataset file(s) at: %s", file_path)
        if os.path.isfile(file_path):
            # mock os.walk return structure
            files = [(os.path.dirname(file_path), [], [file_path])]
        else:
            files = list(os.walk(file_path))

        for root, _, data_files in tqdm.tqdm(files):
            for data_file in data_files:
                if not data_file.endswith("conll"):
                    continue
                # TK TODO REMOVE - the language should be passed in configuration
                lang = data_file.split('/')[-2]
                logger.info("language {}".format(lang))

                with codecs.open(os.path.join(root, data_file), 'r', encoding='utf8') as open_file:
                    for line in open_file:
                        line = line.strip()
                        if line == '' or line.startswith("#"):
                            # Conll format data begins and ends with lines containing a hash,
                            # which may or may not occur after an empty line. To deal with this
                            # we check if the sentence is empty or not and if it is, we just skip
                            # adding instances, because there aren't any to add.
                            if not sentence:
                                continue
                            cur_instances = self._process_sentence(sentence,
                                                                   predicates,
                                                                   senses,
                                                                   predicate_argument_labels,
                                                                   sentence_count)
                            instances.extend(cur_instances)
                            try:
                                assert instances[-1].sentence_id == sentence_count or len(cur_instances) == 0
                            except:
                                print("Problem with instance/sentence_id in allennlp/data/dataset_readers/semantic_role_labeling_conll09.py")
                                ipy.embed()
                            sentence_count += 1

                            # Reset everything for the next sentence.
                            sentence = []
                            predicates = []
                            senses = []
                            predicate_argument_labels = []
                            continue

                        conll_components = line.split()
                        word = conll_components[1]
                        if not (len(word) > 3 and word[:3] == lang and word[3] == ':'):
                            word = lang + ':' + word
                        sentence.append(word)
                        word_index = len(sentence) - 1
                        if word_index == 0:
                            # We're starting a new sentence. Here we set up a list of lists
                            # for the I/O labels for the annotation for each predicate.
                            predicate_argument_labels = [[] for _ in conll_components[14:]]

                        num_annotations = len(predicate_argument_labels)
                        is_predicate = False
                        # Iterate over all predicate annotations for the current sentence.
                        for annotation_index in range(num_annotations):
                            annotation = conll_components[14 + annotation_index]
                            label = annotation.strip("()*")

                            """
                            TK TODO REMOVE
                            This could be replaced by just one line:
                            predicate_argument_labels[annotation_index].append(label)
                            where '_' would be used as the "no annotation" symbol. But in
                            case there is downstream code looking for B/I/O tags, use them
                            for now.
                            """
                            if "_" in annotation:
                                # This word isn't an argument for this predicate.
                                predicate_argument_labels[annotation_index].append("O")
                            else:
                                # The word is an arg with a particular semantic role label.
                                # Append the label to the 'predicate_argument_labels' list
                                # for the current predicate (indexed by 'annotation_index')
                                bio_label = "B-" + label
                                predicate_argument_labels[annotation_index].append(bio_label)

                        # If this word is as a predicate, we need to record its index.
                        # This also has the side effect of ordering the predicates by 
                        # their location in the sentence, automatically aligning them 
                        # with the annotations.
                        # TK TODO TOFIX REMOVE ETC
                        # This assumes gold pred ID, though. If there was a "predicted pred ID column"
                        # we should use that instead.
                        if conll_components[12] == "Y":
                            predicates.append(word_index)
                            senses.append(conll_components[13])
                            token_to_senses[word].add(conll_components[13])

        for instance in instances:
            senses = []
            if 1 in instance.fields['pred_indicator'].labels:
                pred_ind = instance.fields['pred_indicator'].labels.index(1)
                pred_token = instance.fields['tokens'].tokens[pred_ind]
                for pred_sense in token_to_senses[pred_token.text]:
                    senses.append(LabelField(pred_sense, label_namespace='sense_labels'))
            else:
                senses.append(LabelField('_', label_namespace='sense_labels')) #hopefully never relevant
            instance.fields['pred_sense_set'] = ListField(senses)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        logger.info("Read %d SRL instances from %d sentences in dataset file(s) at: %s", len(instances), sentence_count, file_path)
        return Dataset(instances)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         pred_label: List[int],
                         pred_sense: str,
                         tags: List[str] = None, sentence_id=None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a predicate label.  The predicate label 
        should be a one-hot binary vector, the same length as the tokens, indicating the position
        of the predicate to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['pred_indicator'] = SequenceLabelField(pred_label, text_field)
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)

        # for predicate sense disambiguation
        # use name *_labels because we don't want an UNK or padding
        fields['pred_sense'] = LabelField(pred_sense, label_namespace='sense_labels')

        inst = Instance(fields)
        if sentence_id is not None:
            inst.sentence_id = sentence_id
        else:
            print("Problem with instance/sentence_id in allennlp/data/dataset_readers/semantic_role_labeling_conll09.py")
            ipy.embed()
        return inst

    @classmethod
    def from_params(cls, params: Params) -> 'SrlReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return SrlReader(token_indexers=token_indexers)
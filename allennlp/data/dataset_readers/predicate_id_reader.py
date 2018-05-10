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
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField, ListField, MapLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

import IPython as ipy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("predid")
class PredicateIdReader(DatasetReader):
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
    A ``Dataset`` of ``Instances`` for Predicate Identification in semantic role labeling.

    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, 
                 languages: List[str] = None, 
                 for_training: bool=True) -> None:
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(), "pos_tags": SingleIdTokenIndexer()}
        self.languages = languages
        self.for_training = for_training
        self.include_empty = True

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          sentence_lemmas: List[str],
                          sentence_pos: List[str],
                          predicate_indices: List[int],
                          sentence_id: int) -> Instance:
        """
        Parameters
        ----------
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        sentence_lemmas : ``List[str]``, required.
            The lemmatized sentence.
        sentence_pos : ``List[str]``, required.
            The part-of-speech tags for each word in the sentence.
        predicate_indices : ``List[int]``, required.
            The indexes of the predicates (both verbal and nominal) in the
            sentence which have an associated annotation.
        sentence_id : ``List[str]``, required.
            A unique identifier for the sentence, so that instances belonging 
            to the same sentence can be connected.

        Returns
        -------
        An Instance.

        """
        tokens = [Token(text=t, pos=p) for t,p in zip(sentence_tokens,sentence_pos)]
        # to include lemma, need to subclass Token and add another token_indexer maybe? (or maybe it can be the same as the tokens one, using word vectors)
        if not predicate_indices and not self.include_empty:
            # If we're not interested in unannotated sentences, throw this instance away
            return None
        pred_labels = [0 for _ in sentence_tokens]
        for pred_index in predicate_indices:
            pred_labels[pred_index] = 1
        instance = self.text_to_instance(tokens, pred_labels, sentence_id)
        return instance


    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         pred_label: List[int],
                         sentence_id=None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a predicate label.  The predicate label 
        should be a one-hot binary vector, the same length as the tokens, indicating the position
        of the predicate to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        if self.for_training:
            fields['pred_indicator'] = SequenceLabelField(pred_label, text_field)
        fields['metadata'] = MetadataField({'sentence_id':sentence_id})
        inst = Instance(fields)
        return inst

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        lang_instances = {}
        sentence_count = 0

        logger.info("Reading instances from dataset file(s) at: %s", file_path)
        if self.languages is not None:
            logger.info("Languages: " + ", ".join(self.languages))
        if os.path.isfile(file_path):
            # mock os.walk return structure
            files = [(os.path.dirname(file_path), [], [file_path])]
        else:
            files = list(os.walk(file_path))

        for root, _, data_files in files:
            lang = os.path.basename(root.strip('/'))
            if len(data_files) == 0:
                continue
            elif self.languages is not None and lang not in self.languages:
                logger.info("skipping language {} which was not in config".format(lang))
                continue
            else:
                logger.info("processing files for language {}".format(lang))
            instances = []
            for data_file in data_files:
                if not data_file.endswith("conll"):
                    logger.info("skipping file {} which does not end in .conll".format(data_file))
                    continue
                if not os.path.isfile(os.path.join(root, data_file)):
                    logger.info("skipping file {} which could not be found".format(data_file))
                    continue
                instances += self.read_file(os.path.join(root, data_file), sentence_count)
            lang_instances[lang] = instances

        instances = self.balance_by_instances(lang_instances)
        

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        logger.info("Read %d SRL instances from %d sentences in dataset file(s) at: %s", len(instances), sentence_count, file_path)
        return Dataset(instances)

        
    def read_file(self, filename, sc):
        instances = []
        sentence = []
        lemmas = []
        pos_tags = []
        predicates = []
        sentence_count = 0

        for line in codecs.open(filename, 'r', encoding='utf8'):
            line = line.strip()
            if line == "" or line.startswith("#"):
                # Conll format data (sometimes) begins and ends with lines containing a hash,
                # which may or may not occur after an empty line. To deal with this
                # we check if the sentence is empty or not and if it is, we just skip
                # adding instances, because there aren't any to add.
                if not sentence:
                    continue
                cur_instance = self._process_sentence(sentence,
                                                       lemmas,
                                                       pos_tags,
                                                       predicates,
                                                       sentence_count)
                if cur_instance is not None:
                    instances.append(cur_instance)
                sentence_count += 1

                # Reset everything for the next sentence.
                sentence = []
                lemmas = []
                pos_tags = []
                predicates = []
                continue

            conll_components = line.split()
            word = conll_components[1] # 1 for surface form, 3 for predicted lemma
            lemma = conll_components[3]
            if not (len(word) > 3 and word[3] == ':'):
                prefix = lang
                if '_' in prefix:
                    prefix = prefix.split('_')[0]
                word = prefix + ':' + word
            sentence.append(word)
            lemmas.append(lemma)
            pos_tags.append(conll_components[5]) # use predicted part of speech
            word_index = len(sentence) - 1

            # If this word is as a predicate, we need to record its index.
            # This also has the side effect of ordering the predicates by 
            # their location in the sentence, automatically aligning them 
            # with the annotations.
            if conll_components[12] == "Y":
                predicates.append(word_index)

        return instances


    def balance_by_instances(self,lang_instances):
        instances = []
        max_lang = max(lang_instances.keys(), key=lambda lang: len(lang_instances[lang]))
        max_len = len(lang_instances[max_lang])
        for lang in lang_instances:
            if len(lang_instances[lang]) < max_len:
                factor = int(max_len/len(lang_instances[lang]))
                lang_instances[lang] = lang_instances[lang] * factor
                diff = max_len - len(lang_instances[lang])
                lang_instances[lang] += lang_instances[lang][:diff]
            assert len(lang_instances[lang]) == max_len
            instances += lang_instances[lang]
        return instances


    def balance_by_sentences(self,lang_instances):
        instances = []
        max_lang = max(lang_instances.keys(), key=lambda lang: len(set([sid for instance in lang_instances[lang]])))
        max_len = len(lang_instances[max_lang])
        for lang in lang_instances:
            if len(lang_instances[lang]) < max_len:
                factor = int(max_len/len(lang_instances[lang]))
                lang_instances[lang] = lang_instances[lang] * factor
                diff = max_len - len(lang_instances[lang])
                lang_instances[lang] += lang_instances[lang][:diff]
            assert len(lang_instances[lang]) == max_len
            instances += lang_instances[lang]
        return instances


    def balance_by_sum(self,lang_instances):
        instances = []
        for lang in lang_instances:
            instances += lang_instances[lang]
        return instances

    @classmethod
    def from_params(cls, params: Params) -> 'PredicateIdReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lps = params.pop("languages")
        languages = list(lps.values())
        params.assert_empty(cls.__name__)
        return PredicateIdReader(token_indexers=token_indexers, languages=languages)

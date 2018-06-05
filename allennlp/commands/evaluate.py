"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: run [command] evaluate [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import Dict, Any, TextIO
import argparse
import logging
import tqdm
import os
from collections import defaultdict

import torch

from allennlp.common.util import prepare_environment
from allennlp.data import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables

from allennlp.training.metrics.external_conll_eval import write_to_conll_2009_eval_file, write_predicates_to_conll_file
import IPython as ipy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Evaluate the specified model + dataset'''
    subparser = parser.add_parser(
            'evaluate', description=description, help='Evaluate the specified model + dataset')
    subparser.add_argument('--archive_file',
                           type=str,
                           required=True,
                           help='path to an archived trained model')
    subparser.add_argument('--evaluation_data_file',
                           type=str,
                           required=True,
                           help='path to the file containing the evaluation data')
    subparser.add_argument('--cuda_device',
                           type=int,
                           default=-1,
                           help='id of GPU to use (if any)')
    subparser.add_argument('--print_predictions',
                           type=str,
                           help='print CoNLL files containing the predicted and labelings to the given path')
    subparser.add_argument('--print_predicates',
                           type=str,
                           help='print CoNLL files containing the predicted predicates (with no labels) to the given path')
    subparser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')

    subparser.set_defaults(func=evaluate_from_args)

    return subparser


def evaluate(model: Model,
             dataset: Dataset,
             iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    model.eval()

    generator = iterator(dataset, num_epochs=1)
    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset))
    for batch in generator:
        tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
        print("Run model forward on batch")
        model.forward(**tensor_batch) # stores TP/FN counts for get_metrics as a side-effect
        metrics = model.get_metrics()
        description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
        generator_tqdm.set_description(description)

    return model.get_metrics(reset=True)


def predict_conll(model: Model,
                  dataset: Dataset,
                  iterator: DataIterator,
                  cuda_device: int,
                  predict_file: TextIO,
                  gold_file: TextIO) -> Dict[str, Any]:
    model.eval() #sets the model to evaluation mode--no dropout, batchnorm, other stuff?

    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(dataset.instances, total=len(dataset.instances))

    # Recompile instances into sentences (each instance has only one predicate, but
    #   multiple instances come from a single sentence and should be printed thus)
    # Map sentence indices to values
    all_words = {}
    all_predicate_inds = defaultdict(list)
    all_gold_senses = defaultdict(list)
    all_predicted_senses = defaultdict(list)
    all_pos_tags = defaultdict(list)
    all_gold_tags = defaultdict(list)
    all_predicted_tags = defaultdict(list)

    print("setting up conll output")

    for idx, instance in enumerate(generator_tqdm):
        output = model.forward_on_instance(instance, cuda_device, calculate_loss=False)
        predicted_tags = output['tags']
        pos_tags = instance.pos_tags
        gold_senses = instance.fields['pred_sense'].label
        gold_tags = instance.fields['tags'].labels
        tokens = instance.fields['tokens'].tokens
        words = [t.text for t in tokens]
        pred_indices = instance.fields['pred_indicator'].labels
        
        sense_probabilities = output['psd_probabilities']
        
        predicted_sense = output['sense']
        if predicted_sense == model.vocab._oov_token:
            # not a real predicate sense, because we didn't recognize the predicate
            # guess it with a heuristic
            tok_lemma = instance.fields['pred_sense_set'].index_label
            predicted_sense = tok_lemma.split(':')[-1]
            #predicted_sense += '.01'
        if hasattr(instance,'sentence_id'):
            sid = instance.sentence_id
        else:
            sid = instance.fields["metadata"].metadata["sentence_id"]
        if sid in all_words:
            assert all_words[sid] == words
        else:
            all_words[sid] = words
        all_predicate_inds[sid].append(pred_indices)
        all_gold_senses[sid].append(gold_senses)
        all_predicted_senses[sid].append(predicted_sense)
        all_gold_tags[sid].append(gold_tags)
        all_predicted_tags[sid].append(predicted_tags)
        all_pos_tags[sid] = pos_tags

    for sid in all_words:
        write_to_conll_2009_eval_file(predict_file, gold_file,
                                      all_words[sid],
                                      all_predicate_inds[sid],
                                      all_gold_senses[sid],
                                      all_predicted_senses[sid],
                                      all_gold_tags[sid],
                                      all_predicted_tags[sid],
                                      all_pos_tags[sid])
    print("printed conll output")

    return True


def predict_predicates(model: Model,
                     dataset: Dataset,
                     iterator: DataIterator,
                     cuda_device: int,
                     predict_file: TextIO,
                     gold_file: TextIO) -> Dict[str, Any]:
    model.eval() #sets the model to evaluation mode--no dropout, batchnorm, other stuff?

    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(dataset.instances, total=len(dataset.instances))

    print("setting up conll output")
    
    all_sids = {}
    for idx, instance in enumerate(generator_tqdm):
        output = model.forward_on_instance(instance, cuda_device, calculate_loss=False)
        pred_indices = output['tags']
        sid = instance.fields['metadata'].metadata['sentence_id']
        if sid in all_sids:
            if pred_indices != all_sids[sid][0]:
                print("Different results for sentence {}".format(sid))
                ipy.embed()
        all_sids[sid] = (instance, output)
        tokens = instance.fields['tokens'].tokens
        words = [t.text for t in tokens]
        pos = [t.pos_ for t in tokens]
        gold_preds = instance.fields['pred_indicator'].labels
        write_predicates_to_conll_file(predict_file, gold_file,
                                       words,
                                       pos,
                                       pred_indices, gold_preds)
    print("printed conll output")

    return True


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    torch.cuda.set_device(args.cuda_device)
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.cuda(args.cuda_device)
    model.eval()

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    #dataset_reader.for_training = False
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model.vocab)

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics = evaluate(model, dataset, iterator, args.cuda_device)

    if args.print_predictions:
        directory = os.path.dirname(args.archive_file)
        predict_filename = args.print_predictions
        predict_file = open(predict_filename, 'w')
        gold_file = open(os.path.join(directory, 'gold.conll'), 'w')
        predictions = predict_conll(model, dataset, iterator, args.cuda_device, predict_file, gold_file)
    if args.print_predicates:
        directory = os.path.dirname(args.archive_file)
        predict_filename = args.print_predicates
        predict_file = open(predict_filename, 'w')
        gold_file = open(os.path.join(directory, 'gold.conll'), 'w')
        predictions = predict_predicates(model, dataset, iterator, args.cuda_device, predict_file, gold_file)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics

from typing import Dict

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TextFieldEmbedder.register("optional")
class OptionalTextFieldEmbedder(BasicTextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary order.
    """

    @overrides
    def forward(self, text_field_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_representations = []
        keys = sorted(text_field_input.keys())
        for key in keys:
            if key not in self._token_embedders:
                continue
            tensor = text_field_input[key]
            embedder = self._token_embedders[key]
            token_vectors = embedder(tensor)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'OptionalTextFieldEmbedder':
        token_embedders = {}
        keys = list(params.keys())
        for key in keys:
            embedder_params = params.pop(key)
            if ("embedding_dim" in embedder_params and \
                embedder_params.get("embedding_dim") == 0):
                # skip 0-dimensional embedders
                continue
            token_embedders[key] = TokenEmbedder.from_params(vocab, embedder_params)
        params.assert_empty(cls.__name__)
        return cls(token_embedders)

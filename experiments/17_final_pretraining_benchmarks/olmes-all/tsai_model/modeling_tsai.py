import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation.utils import GenerationMixin

# Force inclusion of local auxiliary modules. Use relative imports
from .liger_ops import LigerSwiGLUMLP
from .reversible_ops_midpoint import ReversibleMidpointStack

from .recurrence_model_1b import Model1B, ModelConfig


class TsaiConfig(PretrainedConfig):
    model_type = "tsai"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        num_layers=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # HF compatibility
        self.num_hidden_layers = num_layers

        self.is_decoder = True


class TsaiForCausalLM(PreTrainedModel, GenerationMixin):

    config_class = TsaiConfig

    # HuggingFace compatibility
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)

        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        # Training architecture supports Kronecker embeddings that require tokenizer codec metadata.
        # For evaluation we bypass that mechanism and use a standard embedding table compatible with HF pipelines.
        self.model = Model1B(model_config, embedding_type="standard")

    def get_input_embeddings(self):
        return self.model.token_embed

    def set_input_embeddings(self, new_embeddings):
        self.model.token_embed = new_embeddings

    def get_output_embeddings(self):
        return self.model.token_embed

    def forward(self, input_ids, attention_mask=None, **kwargs):

        logits_ntp, logits_mtp = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_memory=False
        )

        return CausalLMOutput(logits=logits_ntp)
from d2f_engine.config import Config
from d2f_engine.utils.loader import load_model
from d2f_engine.models.dream import DreamForDiffusionLM
from d2f_engine.models.qwen3 import Qwen3ForCausalLM
from d2f_engine.models.llada import LLaDAForDiffusionLM


class AutoModelLM:
    MODEL_MAPPING = {
        "qwen3": Qwen3ForCausalLM,
        "dream": DreamForDiffusionLM,
        "llada": LLaDAForDiffusionLM,
        "llada-1.5": None,
        "dream-on": None,
        "sdar": None,
    }
    @classmethod
    def from_config(cls, config: Config):
        model = cls.MODEL_MAPPING[config.model_name](config.hf_config)
        return load_model(model, config)
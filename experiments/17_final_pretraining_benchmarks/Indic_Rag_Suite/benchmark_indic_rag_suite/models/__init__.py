# Models are registered on import
from benchmark_indic_rag_suite.models import hf_backends  # noqa: F401
from benchmark_indic_rag_suite.models import small  # noqa: F401
from benchmark_indic_rag_suite.models.registry import get_generation_model, get_retrieval_model

__all__ = ["get_retrieval_model", "get_generation_model"]

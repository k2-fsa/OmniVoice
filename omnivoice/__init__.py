import warnings
from importlib.metadata import PackageNotFoundError, version

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)

try:
    __version__ = version("omnivoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "OmniVoice",
    "OmniVoiceConfig",
    "OmniVoiceGenerationConfig",
    "VoiceClonePrompt",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        import torchaudio  # noqa: F401
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Loading OmniVoice model classes currently requires torchaudio "
            "through the Transformers Higgs audio tokenizer. Install the "
            "optional extra with `pip install 'omnivoice[audio-tools]'`."
        ) from exc

    from omnivoice.models import omnivoice as model_module

    value = getattr(model_module, name)
    globals()[name] = value
    return value

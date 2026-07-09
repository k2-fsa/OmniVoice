import warnings
from importlib.metadata import PackageNotFoundError, version

from omnivoice._offline import configure_offline_defaults

configure_offline_defaults()

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

__all__ = ["OmniVoice", "OmniVoiceConfig", "OmniVoiceGenerationConfig"]


def __getattr__(name):
    if name in __all__:
        from omnivoice.models.omnivoice import (
            OmniVoice,
            OmniVoiceConfig,
            OmniVoiceGenerationConfig,
        )

        values = {
            "OmniVoice": OmniVoice,
            "OmniVoiceConfig": OmniVoiceConfig,
            "OmniVoiceGenerationConfig": OmniVoiceGenerationConfig,
        }
        return values[name]
    raise AttributeError(f"module 'omnivoice' has no attribute {name!r}")

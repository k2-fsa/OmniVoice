"""Local audiobook planning tools for OmniVoice."""

from omnivoice.audiobook.docx import DocxDocument, DocxParagraph, extract_docx_structure
from omnivoice.audiobook.chunking import AudiobookChunk, ChunkingConfig, chunk_docx_document
from omnivoice.audiobook.generation import AudiobookGenerationJob, SegmentWorkItem
from omnivoice.audiobook.generation import load_generation_checkpoint, write_generation_checkpoint
from omnivoice.audiobook.mastering import MasteringOptions, concat_audio_files, remaster_audio
from omnivoice.audiobook.openrouter import OpenRouterAudiobookClient, OpenRouterConfig
from omnivoice.audiobook.planner import (
    AudiobookPlanConfig,
    create_audiobook_plan,
    create_audiobook_plan_from_docx,
    create_audiobook_plan_from_openrouter_results,
    plan_to_json,
)
from omnivoice.audiobook.schema import (
    AudiobookChapter,
    AudiobookPlan,
    AudiobookProject,
    AudiobookQcTargets,
    AudiobookSegment,
    VoiceProfile,
)

__all__ = [
    "AudiobookChapter",
    "AudiobookChunk",
    "AudiobookGenerationJob",
    "AudiobookPlan",
    "AudiobookPlanConfig",
    "AudiobookProject",
    "AudiobookQcTargets",
    "AudiobookSegment",
    "ChunkingConfig",
    "DocxDocument",
    "DocxParagraph",
    "MasteringOptions",
    "OpenRouterAudiobookClient",
    "OpenRouterConfig",
    "SegmentWorkItem",
    "VoiceProfile",
    "chunk_docx_document",
    "concat_audio_files",
    "create_audiobook_plan",
    "create_audiobook_plan_from_docx",
    "create_audiobook_plan_from_openrouter_results",
    "extract_docx_structure",
    "plan_to_json",
    "remaster_audio",
    "load_generation_checkpoint",
    "write_generation_checkpoint",
]

"""Local audiobook planning tools for OmniVoice."""

from omnivoice.audiobook.docx import DocxDocument, DocxParagraph, extract_docx_structure
from omnivoice.audiobook.planner import (
    AudiobookPlanConfig,
    create_audiobook_plan,
    create_audiobook_plan_from_docx,
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
    "AudiobookPlan",
    "AudiobookPlanConfig",
    "AudiobookProject",
    "AudiobookQcTargets",
    "AudiobookSegment",
    "DocxDocument",
    "DocxParagraph",
    "VoiceProfile",
    "create_audiobook_plan",
    "create_audiobook_plan_from_docx",
    "extract_docx_structure",
    "plan_to_json",
]

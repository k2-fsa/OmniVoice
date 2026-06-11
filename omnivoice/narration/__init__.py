__all__ = [
    "NarrationCache",
    "NarrationGenerationResult",
    "NarrationPlan",
    "NarrationSegment",
    "PausePreset",
    "assemble_from_plan",
    "assemble_segments",
    "audio_duration_seconds",
    "float_to_int16",
    "generate_narration",
    "parse_narration_text",
    "regenerate_segment",
]


def __getattr__(name):
    if name in {"assemble_segments", "audio_duration_seconds", "float_to_int16"}:
        from omnivoice.narration.assembler import (
            assemble_segments,
            audio_duration_seconds,
            float_to_int16,
        )

        return {
            "assemble_segments": assemble_segments,
            "audio_duration_seconds": audio_duration_seconds,
            "float_to_int16": float_to_int16,
        }[name]

    if name == "parse_narration_text":
        from omnivoice.narration.parser import parse_narration_text

        return parse_narration_text

    if name in {"NarrationPlan", "NarrationSegment", "PausePreset"}:
        from omnivoice.narration.schema import NarrationPlan, NarrationSegment, PausePreset

        return {
            "NarrationPlan": NarrationPlan,
            "NarrationSegment": NarrationSegment,
            "PausePreset": PausePreset,
        }[name]

    if name == "NarrationCache":
        from omnivoice.narration.cache import NarrationCache

        return NarrationCache

    if name in {
        "NarrationGenerationResult",
        "assemble_from_plan",
        "generate_narration",
        "regenerate_segment",
    }:
        from omnivoice.narration.generator import (
            NarrationGenerationResult,
            assemble_from_plan,
            generate_narration,
            regenerate_segment,
        )

        return {
            "NarrationGenerationResult": NarrationGenerationResult,
            "assemble_from_plan": assemble_from_plan,
            "generate_narration": generate_narration,
            "regenerate_segment": regenerate_segment,
        }[name]

    raise AttributeError(f"module 'omnivoice.narration' has no attribute {name!r}")

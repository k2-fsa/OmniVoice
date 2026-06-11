# Audiobook Standards

These defaults are internal production standards for OmniVoice audiobook tooling.

## Segmentation

- Target segment size: 120 to 900 characters.
- Split by sentence first, then punctuation if a sentence is too long.
- Preserve author meaning and quoted text.
- Do not remove technical terms, legal text, or dialogue markers without approval.

## Pacing

| Context | Target WPM | Notes |
| --- | ---: | --- |
| Technical book | 135-155 | More pauses around lists, code, formulas, definitions |
| Fiction | 145-170 | More variation for dialogue and scene changes |

## Pauses

| Boundary | Default |
| --- | ---: |
| Comma | 220-300 ms |
| Sentence | 520-750 ms |
| Ellipsis | 800-1100 ms |
| Paragraph | 950-1400 ms |
| Chapter | 1800-2500 ms |

## QC Targets

- Sample rate target: 44100 Hz unless model output requires another rate.
- Peak target: no samples above -3 dBFS after mastering.
- Initial loudness target: -20 LUFS with 2 LU tolerance.
- No zero-byte audio files.
- No missing chunks.
- No hidden clipping or failed joins.

# SQLite Projects

The frontend workspace uses SQLite to preserve all project state required to
continue audiobook work.

SQLite stores project metadata, document references, chunks, plans, provider run
metadata, token/cost records, audio asset indexes, QC reports, checkpoints, and
backup records.

Audio files remain on disk and are referenced by path and hash.

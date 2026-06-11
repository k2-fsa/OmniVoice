import unittest

from omnivoice.audiobook.chunking import ChunkingConfig, chunk_docx_document
from omnivoice.audiobook.docx import DocxDocument, DocxParagraph


class AudiobookChunkingTest(unittest.TestCase):
    def test_chunks_large_docx_by_word_budget(self):
        paragraphs = [
            DocxParagraph(index=i, text=" ".join([f"palavra{i}"] * 120))
            for i in range(15)
        ]
        document = DocxDocument(path="book.docx", sha256="hash", paragraphs=paragraphs)

        chunks = chunk_docx_document(
            document,
            ChunkingConfig(max_words=500, target_words=360, overlap_summary_words=10),
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.word_count <= 500 for chunk in chunks))
        self.assertIsNone(chunks[0].previous_summary)
        self.assertIsNotNone(chunks[1].previous_summary)
        self.assertEqual(chunks[0].paragraph_start, 0)

    def test_splits_oversized_paragraph(self):
        document = DocxDocument(
            path="book.docx",
            sha256="hash",
            paragraphs=[DocxParagraph(index=0, text=" ".join(["x"] * 650))],
        )

        chunks = chunk_docx_document(
            document,
            ChunkingConfig(max_words=300, target_words=300),
        )

        self.assertEqual(len(chunks), 3)
        self.assertIn("oversized_paragraph_split", chunks[0].warnings)


if __name__ == "__main__":
    unittest.main()

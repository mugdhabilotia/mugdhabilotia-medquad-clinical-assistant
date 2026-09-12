#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data Engineering & Chunking Pipeline for NIH MedQuAD (`scripts/prepare_medquad_data.py`).

Fulfills Sprint 2 requirements:
- Parses NIH MedQuAD documents (from XML files or raw JSONL).
- Implements a token-aware sliding window: 500-token chunks with 10% (50-token) overlap.
- Retains critical metadata: `id`, `title`, `question`, `answer`, and `source attribution`.
- Produces ingestion-ready JSONL for Vertex AI Search / Discovery Engine.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))
    def chunk_tokens(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        tokens = _ENCODER.encode(text)
        if len(tokens) <= chunk_size:
            return [text]
        step = chunk_size - overlap
        chunks = []
        for i in range(0, len(tokens), step):
            chunk_toks = tokens[i : i + chunk_size]
            chunks.append(_ENCODER.decode(chunk_toks))
            if i + chunk_size >= len(tokens):
                break
        return chunks
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)
    def chunk_tokens(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        words_per_chunk = int(chunk_size * 0.75)
        overlap_words = int(overlap * 0.75)
        step = words_per_chunk - overlap_words
        if len(words) <= words_per_chunk:
            return [text]
        chunks = []
        for i in range(0, len(words), step):
            chunk_slice = words[i : i + words_per_chunk]
            chunks.append(" ".join(chunk_slice))
            if i + words_per_chunk >= len(words):
                break
        return chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_medquad_data")


def parse_medquad_xml(xml_path: Path) -> list[dict[str, Any]]:
    """Parses a single NIH MedQuAD XML file into structured clinical records."""
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        doc_id = root.attrib.get("id", xml_path.stem)
        url = root.findtext("URL") or ""
        focus = root.findtext("Focus") or ""

        qa_pairs = root.findall(".//QAPair")
        for idx, qa in enumerate(qa_pairs, start=1):
            qid = qa.attrib.get("pid", f"{doc_id}-{idx}")
            question = qa.findtext("Question") or ""
            answer = qa.findtext("Answer") or ""

            if not question and not answer:
                continue

            records.append({
                "id": qid,
                "title": focus,
                "focus": focus,
                "question": question,
                "answer": answer,
                "source": f"NIH MedQuAD / {xml_path.parent.name}",
                "url": url,
            })
    except Exception as exc:
        logger.warning(f"Failed to parse XML file {xml_path}: {exc}")
    return records


def process_records_into_chunks(
    records: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """Applies a 500-token sliding window with 10% (50-token) overlap to answers."""
    ingestion_records = []
    seen_ids = set()

    for rec in records:
        base_id = rec.get("id") or rec.get("_id") or "UNKNOWN"
        title = rec.get("title") or rec.get("focus") or ""
        question = rec.get("question") or ""
        answer = rec.get("answer") or ""
        source = rec.get("source") or "NIH MedQuAD"
        url = rec.get("url") or ""

        full_content = f"Title: {title}\nQuestion: {question}\n\nAnswer: {answer}"
        chunks = chunk_tokens(full_content, chunk_size=chunk_size, overlap=overlap)

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{base_id}-{chunk_idx}" if len(chunks) > 1 else str(base_id)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            b64_content = base64.b64encode(chunk_text.encode("utf-8")).decode("utf-8")

            ingestion_records.append({
                "id": chunk_id,
                "structData": {
                    "title": title,
                    "focus": title,
                    "question": question,
                    "answer": chunk_text,
                    "source": source,
                    "url": url,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "evidence_level": "NIH / Peer-Reviewed Clinical Reference",
                },
                "content": {
                    "mimeType": "text/plain",
                    "rawBytes": b64_content,
                },
            })

    return ingestion_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MedQuAD corpus with 500-token sliding window")
    parser.add_argument("--input", default="/home/admin_/medquad_complete_corpus.jsonl", help="Input XML dir or JSONL file")
    parser.add_argument("--output", default="/home/admin_/medquad_ingestion_ready.jsonl", help="Output JSONL destination")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in tokens (default: 50)")
    args = parser.parse_args()

    input_path = Path(args.input)
    raw_records: list[dict[str, Any]] = []

    if input_path.is_dir():
        logger.info(f"Scanning directory {input_path} for NIH MedQuAD XML files...")
        for xml_file in input_path.glob("**/*.xml"):
            raw_records.extend(parse_medquad_xml(xml_file))
    elif input_path.is_file():
        logger.info(f"Loading existing records from {input_path}...")
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    src_data = data.get("structData", data)
                    raw_records.append({
                        "id": str(data.get("id") or src_data.get("id") or ""),
                        "title": src_data.get("title") or src_data.get("focus") or "",
                        "focus": src_data.get("focus") or src_data.get("title") or "",
                        "question": src_data.get("question") or "",
                        "answer": src_data.get("answer") or "",
                        "source": src_data.get("source") or "NIH MedQuAD",
                        "url": src_data.get("url") or "",
                    })
                except json.JSONDecodeError:
                    continue
    else:
        logger.error(f"Input path {input_path} does not exist.")
        return

    logger.info(f"Extracted {len(raw_records)} source records. Applying 500-token sliding window (overlap={args.overlap})...")
    chunks = process_records_into_chunks(raw_records, chunk_size=args.chunk_size, overlap=args.overlap)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c) + "\n")

    logger.info(f"Saved {len(chunks)} chunked documents to {output_path}.")


if __name__ == "__main__":
    main()

import tempfile
import unittest
from pathlib import Path

from selection_distill.records import read_json, read_jsonl, write_json, write_jsonl


class JsonlRecordTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_and_row_order(self):
        rows = [{"id": "one", "question": "คำถาม"}, {"id": "two", "question": "x?"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)

    def test_malformed_line_reports_its_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id":"ok"}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"rows.jsonl:2"):
                read_jsonl(path)

    def test_json_round_trip_preserves_structured_manifest_values(self):
        value = {"budget": 123, "receipt": None, "complete": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json(path, value)
            self.assertEqual(read_json(path), value)


if __name__ == "__main__":
    unittest.main()

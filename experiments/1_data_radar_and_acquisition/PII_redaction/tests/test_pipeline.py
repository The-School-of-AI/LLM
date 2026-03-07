from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pii_redaction.config import PipelineConfig
from pii_redaction.processor import process_file, process_record, run_pipeline

TMP_ROOT = ROOT / "tests" / ".tmp"
TMP_ROOT.mkdir(exist_ok=True)


class PipelineTests(unittest.TestCase):
    def make_temp_dir(self, name: str) -> Path:
        target = TMP_ROOT / name
        if target.exists():
            for path in sorted(target.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            target.rmdir()
        target.mkdir(parents=True)
        return target

    def test_structured_pii_redaction(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {
                "id": "1",
                "text": "Email jane@example.com phone +1 415 555 0101 ssn 123-45-6789 card 4111 1111 1111 1111",
            },
            config,
        )
        self.assertEqual(decision.action, "drop")
        self.assertGreaterEqual(decision.total_entities, 4)

    def test_url_sanitization_preserves_non_sensitive_params(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {
                "id": "2",
                "text": "Visit https://example.com/reset?email=a%40b.com&lang=en&token=abc",
            },
            config,
        )
        self.assertEqual(decision.action, "keep")
        text = decision.record["text"]
        self.assertIn("lang=en", text)
        self.assertIn("<SENSITIVE_VALUE>", text)
        self.assertNotIn("abc", text)

    def test_url_sanitization_is_case_insensitive_and_handles_repeated_params(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {
                "id": "2b",
                "text": "Visit https://example.com/reset?Token=abc&email=a%40b.com&token=def&lang=en",
            },
            config,
        )
        text = decision.record["text"]
        self.assertEqual(text.count("<SENSITIVE_VALUE>"), 3)
        self.assertIn("lang=en", text)
        self.assertNotIn("abc", text)
        self.assertNotIn("def", text)

    def test_public_figure_name_not_redacted_without_anchor(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "3", "text": "Barack Obama met Satya Nadella in public."},
            config,
        )
        self.assertEqual(decision.total_entities, 0)
        self.assertEqual(decision.record["text"], "Barack Obama met Satya Nadella in public.")

    def test_anchored_name_redacted(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "4", "text": "customer: Alice Smith account no 7788990011"},
            config,
        )
        self.assertEqual(decision.action, "keep")
        self.assertIn("<PERSON_NAME>", decision.record["text"])
        self.assertIn("<ACCOUNT_NUMBER>", decision.record["text"])

    def test_hindi_anchor_and_unicode_name_are_redacted(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "4b", "text": "\u0928\u093e\u092e: \u0930\u0935\u093f \u0915\u0941\u092e\u093e\u0930 \u0906\u0927\u093e\u0930 1234 5678 9123"},
            config,
        )
        self.assertEqual(decision.entities_by_type["PERSON_NAME"], 1)
        self.assertEqual(decision.entities_by_type["AADHAAR_NUMBER"], 1)
        self.assertIn("<PERSON_NAME>", decision.record["text"])

    def test_english_anchor_can_capture_unicode_name(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "4c", "text": "Customer name: \u0905\u0928\u0928\u094d\u092f\u093e \u0926\u0924\u094d\u0924\u093e email anya@example.com"},
            config,
        )
        self.assertEqual(decision.entities_by_type["PERSON_NAME"], 1)
        self.assertEqual(decision.entities_by_type["EMAIL_ADDRESS"], 1)

    def test_hindi_address_anchor_with_romanized_suffix_is_redacted(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "4d", "text": "\u092a\u0924\u093e: 221B Baker Street, Delhi"},
            config,
        )
        self.assertEqual(decision.entities_by_type["PHYSICAL_ADDRESS"], 1)
        self.assertIn("<PHYSICAL_ADDRESS>", decision.record["text"])

    def test_multilingual_structured_detection(self) -> None:
        config = PipelineConfig()
        text = "Aadhaar 1234 5678 9123 and PAN ABCDE1234F and phone +91 98765 43210"
        decision = process_record({"id": "5", "lang": "hi", "text": text}, config)
        self.assertEqual(decision.action, "drop")
        entities = decision.entities_by_type
        self.assertEqual(entities["AADHAAR_NUMBER"], 1)
        self.assertEqual(entities["PAN_NUMBER"], 1)
        self.assertEqual(entities["PHONE_NUMBER"], 1)

    def test_account_number_not_mislabeled_as_aadhaar(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "6", "text": "account number 009988776655 should be redacted as an account only"},
            config,
        )
        self.assertEqual(decision.entities_by_type["ACCOUNT_NUMBER"], 1)
        self.assertNotIn("AADHAAR_NUMBER", decision.entities_by_type)
        self.assertNotIn("PHONE_NUMBER", decision.entities_by_type)
        self.assertIn("<ACCOUNT_NUMBER>", decision.record["text"])

    def test_iban_is_redacted_as_account_number(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "iban-1", "text": "Payment should go to DE89370400440532013000 immediately."},
            config,
        )
        self.assertEqual(decision.entities_by_type["ACCOUNT_NUMBER"], 1)
        self.assertIn("<ACCOUNT_NUMBER>", decision.record["text"])

    def test_invalid_ipv4_is_not_redacted(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "ip-1", "text": "Example placeholder 999.999.999.999 should stay as-is."},
            config,
        )
        self.assertEqual(decision.total_entities, 0)

    def test_valid_ipv6_is_redacted(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "ip-2", "text": "Node address is 2001:0db8:85a3:0000:0000:8a2e:0370:7334"},
            config,
        )
        self.assertEqual(decision.entities_by_type["IP_ADDRESS"], 1)

    def test_invalid_card_number_is_not_redacted(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "card-1", "text": "Sequence 4111 1111 1111 1112 is not a valid card."},
            config,
        )
        self.assertEqual(decision.total_entities, 0)

    def test_non_sensitive_url_is_left_intact(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "url-1", "text": "Read https://example.com/docs?lang=en&view=full for details."},
            config,
        )
        self.assertEqual(decision.total_entities, 0)
        self.assertIn("https://example.com/docs?lang=en&view=full", decision.record["text"])

    def test_nested_text_field_path_is_supported(self) -> None:
        config = PipelineConfig()
        config.schema.text_fields = ["payload.body"]
        decision = process_record(
            {"id": "nested-1", "payload": {"body": "Reach me at nested@example.com"}},
            config,
        )
        self.assertEqual(decision.record["payload"]["body"], "Reach me at <EMAIL_ADDRESS>")

    def test_generic_placeholder_mode_is_supported(self) -> None:
        config = PipelineConfig()
        config.redaction.placeholder_style = "generic"
        decision = process_record(
            {"id": "generic-1", "text": "Email jane@example.com"},
            config,
        )
        self.assertIn("[REDACTED]", decision.record["text"])
        self.assertNotIn("<EMAIL_ADDRESS>", decision.record["text"])

    def test_unchanged_doc_gets_status_metadata(self) -> None:
        config = PipelineConfig()
        decision = process_record(
            {"id": "clean-1", "text": "This document is safe and should remain untouched."},
            config,
        )
        self.assertEqual(decision.record["_redaction"]["status"], "unchanged")
        self.assertEqual(decision.record["_redaction"]["entities_by_type"], {})

    def test_run_pipeline_resume_and_artifacts(self) -> None:
        config = PipelineConfig()
        temp_root = self.make_temp_dir("resume")
        input_path = temp_root / "input.jsonl"
        input_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "a", "text": "hello world"}),
                    json.dumps({"id": "b", "text": "email me at a@example.com"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = temp_root / "out"
        manifest = run_pipeline([input_path], output_dir, config, resume=True)
        self.assertEqual(manifest["summary"]["records_seen"], 2)
        self.assertTrue((output_dir / "run_manifest.json").exists())
        shard_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
        self.assertEqual(len(shard_dirs), 1)
        manifest_again = run_pipeline([input_path], output_dir, config, resume=True)
        self.assertEqual(manifest_again["summary"]["records_seen"], 2)

    def test_run_pipeline_multiple_inputs_aggregate(self) -> None:
        config = PipelineConfig()
        temp_root = self.make_temp_dir("multi-input")
        input_a = temp_root / "a.jsonl"
        input_b = temp_root / "b.jsonl"
        input_a.write_text(json.dumps({"id": "a", "text": "email a@example.com"}) + "\n", encoding="utf-8")
        input_b.write_text(json.dumps({"id": "b", "text": "clean text only"}) + "\n", encoding="utf-8")
        manifest = run_pipeline([input_a, input_b], temp_root / "out", config, resume=False)
        self.assertEqual(manifest["summary"]["records_seen"], 2)
        self.assertEqual(manifest["summary"]["records_kept"], 2)
        self.assertEqual(manifest["entities_by_type"]["EMAIL_ADDRESS"], 1)
        self.assertEqual(len(manifest["files"]), 2)

    def test_process_file_outputs_drop_reasons_without_raw_text(self) -> None:
        config = PipelineConfig()
        temp_root = self.make_temp_dir("drop-output")
        input_path = temp_root / "input.jsonl"
        input_path.write_text(
            json.dumps(
                {
                    "id": "drop-doc",
                    "text": "ssn 123-45-6789 aadhaar 1234 5678 9123 card 4111 1111 1111 1111",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        metrics = process_file(input_path, temp_root / "out", config)
        dropped = (temp_root / "out" / "dropped.jsonl").read_text(encoding="utf-8")
        self.assertIn("drop-doc", dropped)
        self.assertNotIn("123-45-6789", dropped)
        self.assertEqual(metrics["summary"]["records_dropped"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from ground_truth_evaluation import _normalize_payload, _strip_code_fences


class GroundTruthEvaluationParsingTests(unittest.TestCase):
    def test_strip_code_fences_extracts_embedded_json_block(self) -> None:
        raw = (
            "The search returned 91,002 Twitter posts that used the hashtag #auspol "
            "during the first quarter of 2023 (January 1 to March 31, 2023).\n\n"
            "```json\n"
            "{\n"
            '  "status": "success",\n'
            '  "answer": {\n'
            '    "value": 91002,\n'
            '    "unit": "posts"\n'
            "  }\n"
            "}\n"
            "```"
        )

        stripped = _strip_code_fences(raw)

        self.assertEqual(
            stripped,
            '{\n  "status": "success",\n  "answer": {\n    "value": 91002,\n    "unit": "posts"\n  }\n}',
        )

    def test_normalize_payload_parses_embedded_json_block(self) -> None:
        raw = (
            "Narrative text before the machine-readable payload.\n\n"
            "```json\n"
            "{\n"
            '  "status": "success",\n'
            '  "answer": {\n'
            '    "value": 91002,\n'
            '    "unit": "posts",\n'
            '    "comment": "Number of Twitter posts with hashtag #auspol during Q1 2023"\n'
            "  }\n"
            "}\n"
            "```"
        )

        normalized = _normalize_payload(raw)

        self.assertIsNone(normalized.parse_error)
        self.assertEqual(normalized.parser_used, "json")
        self.assertEqual(normalized.status, "success")
        self.assertEqual(
            normalized.answer_payload,
            {
                "value": 91002,
                "unit": "posts",
                "comment": "Number of Twitter posts with hashtag #auspol during Q1 2023",
            },
        )


if __name__ == "__main__":
    unittest.main()

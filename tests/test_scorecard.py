import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from app.ipeds_connect.scorecard import sync_scorecard_cache


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


class ScorecardTests(unittest.TestCase):
    def test_requires_an_api_key(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            sync_scorecard_cache("")

    def test_targeted_sync_deduplicates_and_batches_unitids(self):
        requests = []

        def fake_urlopen(request, timeout, context):
            self.assertEqual(timeout, 45)
            self.assertIsNotNone(context)
            query = parse_qs(urlparse(request.full_url).query)
            requests.append(query)
            ids = query["id"][0].split(",")
            return _Response({"results": [{"id": int(value), "school.name": f"School {value}"} for value in ids]})

        with tempfile.TemporaryDirectory() as directory, patch(
            "app.ipeds_connect.scorecard.urllib.request.urlopen", side_effect=fake_urlopen
        ):
            output = Path(directory) / "scorecard.json"
            sync_scorecard_cache("test-key", output, [str(value) for value in range(101)] + ["1"])
            document = json.loads(output.read_text())

        self.assertEqual(len(requests), 3)
        self.assertEqual(len(document["institutions"]), 101)
        self.assertTrue(all("_fields" in query and "_per_page" in query for query in requests))
        self.assertEqual(document["institutions"][0]["unitid"], "0")


if __name__ == "__main__":
    unittest.main()

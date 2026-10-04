import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID

from backend.session_logging import SessionEventLogger


class SessionEventLoggerTests(unittest.TestCase):
    def test_events_are_appended_to_one_file_for_the_session(self):
        session_id = UUID("01234567-89ab-cdef-0123-456789abcdef")
        request_id = UUID("abcdef01-2345-6789-abcd-ef0123456789")
        with tempfile.TemporaryDirectory() as directory:
            event_logger = SessionEventLogger(Path(directory))
            log_path = event_logger.log(
                session_id=session_id,
                request_id=request_id,
                event_type="user_input",
                payload={"message": "Wie viel ist 15 % von 200?"},
            )
            event_logger.log(
                session_id=session_id,
                request_id=request_id,
                event_type="llm_response",
                payload={"thinking": "Interne Überlegung", "content": "30"},
            )

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertRegex(
            log_path.name,
            rf"^session-\d{{8}}T\d{{6}}\.\d{{6}}Z-{session_id}\.jsonl$",
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["payload"]["message"], "Wie viel ist 15 % von 200?")
        self.assertEqual(records[1]["payload"]["thinking"], "Interne Überlegung")
        self.assertEqual(records[1]["payload"]["content"], "30")
        self.assertEqual(records[0]["request_id"], str(request_id))


if __name__ == "__main__":
    unittest.main()

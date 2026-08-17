import json
import logging

import app.logging_config as lc


def test_setup_logging_configures_json_format():
    lc.setup_logging(level="INFO", json_output=True)
    root = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)


def test_request_context_adds_fields(caplog):
    with caplog.at_level(logging.INFO):
        with lc.request_context(request_id="abc123", stage="stt"):
            logging.getLogger("voice-rag").info(
                "test event", extra={"latency_ms": 50}
            )
    assert caplog.records
    record = caplog.records[-1]
    assert getattr(record, "request_id", None) == "abc123"
    assert getattr(record, "stage", None) == "stt"


def test_json_formatter_outputs_valid_json():
    formatter = lc.JsonFormatter()
    record = logging.LogRecord(
        name="voice-rag", level=logging.INFO, pathname="", lineno=0,
        msg="retrieval done", args=(), exc_info=None,
    )
    record.request_id = "xyz789"  # type: ignore[attr-defined]
    record.stage = "retrieval"  # type: ignore[attr-defined]
    record.latency_ms = 120  # type: ignore[attr-defined]
    record.success = True  # type: ignore[attr-defined]
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["request_id"] == "xyz789"
    assert parsed["stage"] == "retrieval"
    assert parsed["latency_ms"] == 120
    assert parsed["success"] is True
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "voice-rag"
    assert parsed["message"] == "retrieval done"


def test_request_context_without_extra():
    with lc.request_context(request_id="only-id"):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        lc.RequestContextFilter().filter(record)
    assert record.request_id == "only-id"

"""Slack-mocked ingestion failure notification tests (issue #986)."""

from unittest.mock import MagicMock, patch

import pytest

from astroml.chat.slack import SlackConfig, SlackIntegration
from astroml.ingestion.service import IngestionService
from astroml.ingestion.state import StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(path=str(tmp_path / "state" / "ingest.json"))


def _fail_on(bad_ledger):
    def process(ledger_id, _payload):
        if ledger_id == bad_ledger:
            raise RuntimeError("horizon timeout")

    return process


def test_slack_webhook_posted_on_failure(store):
    slack = SlackIntegration(SlackConfig(webhook_url="https://hooks.slack.test/abc"))
    service = IngestionService(state_store=store, notifier=slack.send_webhook)

    with patch("astroml.chat.slack.requests.post") as post:
        post.return_value = MagicMock(status_code=200)
        result = service.ingest(start_ledger=1, end_ledger=5, process_fn=_fail_on(3))

    assert result.errors == ["horizon timeout"]
    post.assert_called_once()
    url = post.call_args.args[0]
    text = post.call_args.kwargs["json"]["text"]
    assert url == "https://hooks.slack.test/abc"
    assert "horizon timeout" in text
    assert "2 processed" in text


def test_no_notification_on_success(store):
    notifier = MagicMock()
    service = IngestionService(state_store=store, notifier=notifier)
    result = service.ingest(start_ledger=1, end_ledger=3)
    assert result.errors == []
    notifier.assert_not_called()


def test_notifier_error_is_logged_not_raised(store, caplog):
    notifier = MagicMock(side_effect=ConnectionError("slack down"))
    service = IngestionService(state_store=store, notifier=notifier)
    result = service.ingest(start_ledger=1, end_ledger=2, process_fn=_fail_on(1))
    assert result.errors == ["horizon timeout"]
    notifier.assert_called_once()
    assert any("notifier raised" in r.message for r in caplog.records)


def test_no_notifier_configured_is_noop(store):
    service = IngestionService(state_store=store)
    result = service.ingest(start_ledger=1, end_ledger=2, process_fn=_fail_on(1))
    assert result.errors == ["horizon timeout"]

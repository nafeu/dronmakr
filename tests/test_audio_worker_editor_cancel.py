from unittest.mock import MagicMock, patch

from dronmakr.audio import audio_worker as aw


def test_cancel_editor_worker_returns_false_when_idle():
    with patch.object(aw, "_editor_worker_proc", None):
        assert aw.cancel_editor_worker() is False


def test_cancel_editor_worker_terminates_active_proc():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 4242
    with patch.object(aw, "_editor_worker_proc", proc):
        assert aw.cancel_editor_worker() is True
    proc.terminate.assert_called_once()


def test_editor_worker_status_reports_active_pid():
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 9999
    with patch.object(aw, "_editor_worker_proc", proc):
        status = aw.editor_worker_status()
    assert status == {"active": True, "pid": 9999}

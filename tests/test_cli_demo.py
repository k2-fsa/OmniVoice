import pytest
from omnivoice.cli.demo import handle_import_error


def test_handle_import_error_windows_dll():
    error = OSError(
        "[WinError 4551] An Application Control policy has blocked this file. Error loading 'shm.dll'"
    )
    with pytest.raises(RuntimeError) as exc_info:
        handle_import_error(error)

    assert "WINDOWS SECURITY ERROR" in str(exc_info.value)
    assert "Windows Defender is blocking PyTorch" in str(exc_info.value)


def test_handle_import_error_other_os_error():
    error = OSError("Some other file not found")
    with pytest.raises(OSError) as exc_info:
        handle_import_error(error)

    assert "Some other file not found" in str(exc_info.value)

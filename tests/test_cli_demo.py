import pytest
from omnivoice.utils.win_errors import handle_import_error


def test_handle_import_error_windows_dll():
    error = OSError(
        "[WinError 4551] An Application Control policy has blocked this file. Error loading 'shm.dll'"
    )
    with pytest.raises(RuntimeError) as exc_info:
        handle_import_error(error)

    assert "WINDOWS SECURITY ERROR" in str(exc_info.value)
    assert "Windows is blocking PyTorch" in str(exc_info.value)


def test_handle_import_error_other_os_error():
    error = OSError("Some other file not found")
    with pytest.raises(OSError) as exc_info:
        handle_import_error(error)

    assert "Some other file not found" in str(exc_info.value)


def test_handle_import_error_winerror_126():
    error = OSError("[WinError 126] Error loading 'shm.dll'")
    with pytest.raises(OSError) as exc_info:
        handle_import_error(error)
    assert "WinError 126" in str(exc_info.value)


def test_initial_import_boundary(monkeypatch):
    import sys
    
    # We need to ensure omnivoice is not in sys.modules to test the initial import
    if 'omnivoice' in sys.modules:
        del sys.modules['omnivoice']
        
    original_import = __import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'torch':
            raise OSError("[WinError 4551] An Application Control policy has blocked this file. Error loading 'shm.dll'")
        return original_import(name, *args, **kwargs)
        
    monkeypatch.setattr('builtins.__import__', mock_import)
    
    with pytest.raises(RuntimeError) as exc_info:
        import omnivoice  # This triggers __init__.py, which tries to import torch
        
    assert "WINDOWS SECURITY ERROR" in str(exc_info.value)

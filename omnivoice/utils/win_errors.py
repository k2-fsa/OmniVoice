def handle_import_error(e: OSError) -> None:
    if "WinError 4551" in str(e):
        msg = (
            "\n" + "="*65 + "\n"
            " ❌ WINDOWS SECURITY ERROR: Application Control Policy Blocked DLL\n"
            "="*65 + "\n"
            " Windows is blocking PyTorch from loading 'shm.dll'.\n"
            " This is likely an Application Control policy block (AppLocker/Smart App Control).\n"
            " Antivirus exclusions do not bypass Application Control.\n"
            " Please check your Windows Event Viewer for policy block events\n"
            " and contact your IT administrator to request an exception.\n"
            + "="*65 + "\n"
        )
        raise RuntimeError(msg) from e
    else:
        raise e

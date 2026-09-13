"""OmniVoice Studio — unified tabbed app.

One window: a server control strip (start/stop server/app.py) above three tabs:
  - Generate        : TTS generator (talks to server/app.py on :9192)
  - Ref Clip Maker  : pick the single best reference window from a file
  - Auto Cut & Merge: cut every voiced span out of a long rip and stitch one ref

Launch via the installed "OmniVoice Studio.exe" (installer/launcher.py) or
gui/studio.bat (venv pythonw, no console window).
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent))
import appconfig  # noqa: E402
from server_panel import ServerBar  # noqa: E402
from generate_tab import GenerateTab  # noqa: E402
from refmaker_tab import RefMakerTab  # noqa: E402
from merge_tab import MergeTab  # noqa: E402

if __name__ == "__main__":
    root = tk.Tk()
    root.title(f"{appconfig.APP_NAME} v{appconfig.APP_VERSION}")
    root.geometry("900x760")
    root.minsize(700, 600)

    ServerBar(root).pack(fill="x")
    ttk.Separator(root, orient="horizontal").pack(fill="x")

    nb = ttk.Notebook(root)
    nb.add(GenerateTab(nb), text="Generate")
    nb.add(RefMakerTab(nb), text="Ref Clip Maker")
    nb.add(MergeTab(nb), text="Auto Cut & Merge")
    nb.pack(fill="both", expand=True)

    root.mainloop()

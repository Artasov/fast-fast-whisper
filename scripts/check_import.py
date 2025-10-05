import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import main

    print("OK:", getattr(main, "app", None) is not None, getattr(main, "app", None).title)
except Exception as e:
    print("ERR:", type(e).__name__, str(e))
    sys.exit(1)

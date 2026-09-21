"""Headless entry point with a single-instance port guard."""
import os
import socket
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_LOG = os.path.join(ROOT, "converter-stdout.log")


def port_busy(port: int = 8787) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def main() -> None:
    if port_busy():
        return

    try:
        stream = open(OUT_LOG, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = stream
        sys.stderr = stream
    except OSError:
        pass

    sys.argv = ["core.converter", "--desensitize", "--log", "converter.log"]
    import runpy
    runpy.run_module("core.converter", run_name="__main__")


if __name__ == "__main__":
    main()

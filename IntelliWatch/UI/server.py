"""Small standard-library server for the IntelliWatch single-page UI."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        requested = Path(urlsplit(self.path).path.lstrip("/"))
        candidate = (ROOT / requested).resolve()
        if not candidate.is_relative_to(ROOT) or not candidate.exists() or candidate.is_dir():
            self.path = "/index.html"
        super().do_GET()


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print(f"IntelliWatch is available at http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, AppHandler).serve_forever()

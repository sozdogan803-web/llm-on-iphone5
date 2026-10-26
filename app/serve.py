"""Serves index.html and the sibling ../models folder over the LAN."""
import http.server
import os
import socket
from urllib.parse import urlparse

APP = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(APP, "..", "models")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".appcache": "text/cache-manifest"}

    def __init__(self, *a, **k):
        super().__init__(*a, directory=APP, **k)

    def translate_path(self, path):
        p = urlparse(path).path
        if p.startswith("/models/"):
            return os.path.join(MODELS, os.path.basename(p))
        return super().translate_path(path)

    def end_headers(self):
        # Desktop browsers only expose SharedArrayBuffer to isolated pages; iOS 10 ignores these.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")  # always revalidate (304 if unchanged); the offline copy is the manifest's job
        super().end_headers()


if __name__ == "__main__":
    # A PC can have several interfaces (VPN, virtual switches), so list them all; the phone's is the one on its Wi-Fi subnet.
    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
        print("Open on the phone: http://%s:8000" % ip, flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()

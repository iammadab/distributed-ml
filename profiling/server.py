from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
import webbrowser


def serve_visualizer(html_path: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the generated visualizer until interrupted."""
    root = html_path.resolve().parent.parent
    relative_path = html_path.resolve().relative_to(root).as_posix()
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
        *args,
        directory=str(root),
        **kwargs,
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/{quote(relative_path)}"
    print(f"Memory visualizer: {url}", flush=True)
    print("Press Ctrl-C to stop the server.", flush=True)
    webbrowser.open(url, new=2)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping memory visualizer.", flush=True)
    finally:
        server.server_close()

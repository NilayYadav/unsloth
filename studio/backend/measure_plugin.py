# Disposable A/B measurement plugin (not part of PR 10552).
# Reports what the probe measured on both branches: how many concurrent requests the
# event loop answered while the upload was in flight, and the worst wait one suffered.
import sys


def pytest_collection_finish(session):
    for name, mod in list(sys.modules.items()):
        if not name.endswith("test_rag_upload_event_loop"):
            continue
        inner = mod._upload_then_ping

        async def timed(path, *args, _inner = inner, **kwargs):
            response, served, worst = await _inner(path, *args, **kwargs)
            print(f"MEASURED {path}: concurrent requests answered during the upload = {served}, "
                  f"worst wait = {worst * 1000:.1f} ms", file = sys.stderr, flush = True)
            return response, served, worst

        mod._upload_then_ping = timed

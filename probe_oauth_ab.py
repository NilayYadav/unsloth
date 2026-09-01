"""Implementation-independent A/B probe for unslothai/unsloth PR #10167.

Drives the production entry point core.inference.mcp_client._client(url, headers,
use_oauth=True), which exists identically on unfixed main and on the PR head, pulls
the OAuth provider back off the transport, and inspects the real token request the
SDK would put on the wire under client_secret_basic.

RFC 6749 2.3.1: a client MUST NOT use more than one authentication method per request.
So when the Basic Authorization header is present, client_id must NOT also be in the body.
"""
import asyncio, base64, os, sys, tempfile
from urllib.parse import parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio", "backend"))
os.environ.setdefault("UNSLOTH_STUDIO_HOME", tempfile.mkdtemp())

from mcp.shared.auth import OAuthClientInformationFull  # noqa: E402
from core.inference import mcp_client  # noqa: E402

URL = "https://mcp.notion.com/mcp"
CID, SEC = "probe-client-id", "probe-client-secret"


def provider():
    client = mcp_client._client(URL, None, use_oauth = True)
    transport = client.transport
    auth = getattr(transport, "auth", None)
    if auth is None:
        for attr in ("_auth", "httpx_client_factory"):
            auth = getattr(transport, attr, None)
            if auth is not None:
                break
    assert auth is not None, f"could not reach the OAuth provider on {type(transport).__name__}"
    return auth


def main():
    auth = provider()
    auth.context.client_info = OAuthClientInformationFull(
        client_id = CID,
        client_secret = SEC,
        token_endpoint_auth_method = "client_secret_basic",
        redirect_uris = auth.context.client_metadata.redirect_uris,
    )
    request = asyncio.run(auth._exchange_token_authorization_code("probe-code", "probe-verifier"))
    form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
    header = request.headers.get("Authorization")

    expected = "Basic " + base64.b64encode(f"{CID}:{SEC}".encode()).decode()
    print(f"PROBE token endpoint  : {request.url}")
    print(f"PROBE Authorization   : {'present and correct' if header == expected else repr(header)}")
    print(f"PROBE body keys       : {sorted(form)}")
    print(f"PROBE client_id in body: {'client_id' in form}")

    assert header == expected, "expected HTTP Basic client authentication header"
    if "client_id" in form:
        print("PROBE RESULT: FAIL -- client_id is in the body AND Basic auth is in the header.")
        print("PROBE RESULT: this is the dual client authentication Notion rejects (RFC 6749 2.3.1).")
        return 1
    print("PROBE RESULT: PASS -- Basic auth header only, no client_id in the body.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

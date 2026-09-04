#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""PR 10319 repro: a host whose first resolved address is unroutable.

Drives the real model-facing page fetch against the real network. Nothing is
monkeypatched; the only thing arranged is the network condition, which is what a
broken-IPv6 VPN looks like to getaddrinfo. Two shapes of unroutable are tested:

  reject  the address answers with ICMP admin-prohibited, so connect() fails at once
  drop    the address swallows the SYN, so connect() fails only by timing out
"""

import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))

HOST = os.environ.get("PROBE_HOST", "example.com")
URL = "https://" + HOST + "/"
SIDE = os.environ["SIDE"]
MODE = os.environ["MODE"]
BUDGET = float(os.environ.get("PROBE_TIMEOUT", "20"))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72, flush = True)


banner("1. What the resolver actually returns for " + HOST)
infos = socket.getaddrinfo(HOST, 443, type = socket.SOCK_STREAM)
order = [i[4][0] for i in infos]
for n, (fam, _t, _p, _c, sa) in enumerate(infos):
    print("  [%d] %-8s %s" % (n, "AF_INET6" if fam == socket.AF_INET6 else "AF_INET", sa[0]))
if not order:
    print("HARNESS FAIL: no addresses resolved")
    sys.exit(2)
print("  first address is IPv6:", ":" in order[0])

banner("2. Raw reachability of each address (proves the arranged condition)")
reach = {}
for addr in order:
    fam = socket.AF_INET6 if ":" in addr else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(6)
    t0 = time.time()
    try:
        s.connect((addr, 443))
        reach[addr] = "REACHABLE"
    except OSError as e:
        reach[addr] = "UNREACHABLE (%s)" % (e or type(e).__name__)
    finally:
        s.close()
    print("  %-42s %-38s %.2fs" % (addr, reach[addr], time.time() - t0))

if ":" not in order[0] or not reach[order[0]].startswith("UNREACHABLE"):
    print("\nHARNESS FAIL: the first resolved address is not an unroutable one.")
    sys.exit(2)
if not any(v == "REACHABLE" for v in reach.values()):
    print("\nHARNESS FAIL: no address is reachable at all; nothing to fall back to.")
    sys.exit(2)

banner("3. The studio SSRF validator: which addresses does it hand the fetcher?")
from core.inference.tools import _fetch_page_text, _validate_and_resolve_host

ok, reason, pinned = _validate_and_resolve_host(HOST, 443)
print("  ok     =", ok)
print("  reason =", repr(reason))
print("  pinned =", repr(pinned))
print("  type   =", type(pinned).__name__)

banner("4. The real model-facing fetch: _fetch_page_text(%r, timeout=%g)" % (URL, BUDGET))
print("   (this is the path that sets a wall-clock deadline for the whole fetch)")
t0 = time.time()
out = _fetch_page_text(URL, timeout = BUDGET)
elapsed = time.time() - t0
failed = out.startswith("Failed to fetch URL") or out.startswith("Blocked:")
print("  elapsed = %.2fs" % elapsed)
print("  failed  =", failed)
print("  result  =", repr(out[:200]))

banner("5. Verdict for SIDE=%s MODE=%s" % (SIDE, MODE))
expect_fail = os.environ["EXPECT"] == "fail"
if failed == expect_fail:
    if failed:
        print("AS EXPECTED - the fetch FAILS here.")
        print("  user-facing error: " + out[:160])
    else:
        print("AS EXPECTED - the fetch SUCCEEDS here.")
        print("  walked past the unroutable %s and read %d chars in %.2fs"
              % (order[0], len(out), elapsed))
    sys.exit(0)
print("MISMATCH: expected the fetch to %s, but it did not." % ("fail" if expect_fail else "succeed"))
print("  result: " + out[:200])
sys.exit(1)

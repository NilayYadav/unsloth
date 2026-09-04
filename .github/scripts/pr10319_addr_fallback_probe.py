#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""PR 10319 repro: a host whose first resolved address is unroutable.

Runs the real studio web-fetch path against the real network. No monkeypatching:
the only thing arranged is the network condition (a configured but blackholed
IPv6 route), which is what a broken-IPv6 VPN looks like to getaddrinfo.
"""

import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))

HOST = os.environ.get("PROBE_HOST", "example.com")
URL = "https://" + HOST + "/"
SIDE = os.environ["SIDE"]


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70, flush = True)


banner("1. What the resolver actually returns for " + HOST)
infos = socket.getaddrinfo(HOST, 443, type = socket.SOCK_STREAM)
order = [i[4][0] for i in infos]
for n, (fam, _t, _p, _c, sa) in enumerate(infos):
    print("  [%d] %-8s %s" % (n, "AF_INET6" if fam == socket.AF_INET6 else "AF_INET", sa[0]))
if not order:
    print("HARNESS FAIL: no addresses resolved")
    sys.exit(2)
first_is_v6 = ":" in order[0]
print("  first address is IPv6:", first_is_v6)

banner("2. Raw reachability of each address (proves the arranged condition)")
reach = {}
for addr in order:
    fam = socket.AF_INET6 if ":" in addr else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(8)
    t0 = time.time()
    try:
        s.connect((addr, 443))
        reach[addr] = "REACHABLE"
    except OSError as e:
        reach[addr] = "UNREACHABLE (%s)" % e
    finally:
        s.close()
    print("  %-42s %-40s %.2fs" % (addr, reach[addr], time.time() - t0))

if not first_is_v6 or not reach[order[0]].startswith("UNREACHABLE"):
    print("\nHARNESS FAIL: the first resolved address is not an unroutable one.")
    print("The condition under test was not established; this run proves nothing.")
    sys.exit(2)
if not any(v == "REACHABLE" for v in reach.values()):
    print("\nHARNESS FAIL: no address is reachable at all; nothing to fall back to.")
    sys.exit(2)

banner("3. The studio SSRF validator (which addresses does it hand the fetcher?)")
from core.inference.tools import _fetch_url_raw, _validate_and_resolve_host

ok, reason, pinned = _validate_and_resolve_host(HOST, 443)
print("  ok      =", ok)
print("  reason  =", repr(reason))
print("  pinned  =", repr(pinned))
print("  type    =", type(pinned).__name__)

banner("4. The real fetch: _fetch_url_raw(%r)" % URL)
t0 = time.time()
err, body, ctype = _fetch_url_raw(URL, timeout = 30)
elapsed = time.time() - t0
print("  error        =", repr(err))
print("  content_type =", repr(ctype))
print("  body bytes   =", len(body))
print("  elapsed      = %.2fs" % elapsed)
if body:
    print("  body head    =", repr(body[:160]))

banner("5. Verdict for SIDE=" + SIDE)
if SIDE == "before":
    if err is None and body:
        print("UNEXPECTED PASS: main fetched the page; the bug did not reproduce.")
        sys.exit(1)
    print("REPRO CONFIRMED: on main the fetch fails outright.")
    print("  user-facing error: " + str(err))
    sys.exit(0)
else:
    if err is not None or not body:
        print("FIX FAILED: with the PR applied the fetch still fails: " + str(err))
        sys.exit(1)
    print("FIX CONFIRMED: the PR walked past the unroutable %s" % order[0])
    print("  and fetched %d bytes of %s from %s" % (len(body), ctype, order[-1]))
    sys.exit(0)

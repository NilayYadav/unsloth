// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

fn fake_system_proxy(port: u16) -> Receiver<String> {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind the fake system proxy");
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { continue };
            let tx = tx.clone();
            std::thread::spawn(move || {
                conn.set_read_timeout(Some(Duration::from_secs(10))).ok();
                let mut reader = BufReader::new(conn.try_clone().unwrap());
                let mut first = String::new();
                if reader.read_line(&mut first).is_err() {
                    return;
                }
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let first = first.trim_end().to_string();
                let _ = tx.send(first.clone());
                let answer: &[u8] = if first.starts_with("CONNECT ") {
                    b"HTTP/1.1 200 Connection established\r\n\r\n"
                } else {
                    b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nproxy-ok"
                };
                let _ = conn.write_all(answer);
                std::thread::sleep(Duration::from_millis(300));
            });
        }
    });
    rx
}

fn ask(page_proxy: SocketAddr, request: &str) -> (String, u128) {
    let started = Instant::now();
    let mut client = TcpStream::connect(page_proxy).expect("reach the page proxy");
    client.set_read_timeout(Some(Duration::from_secs(45))).unwrap();
    client.write_all(request.as_bytes()).unwrap();
    let mut buf = [0u8; 256];
    let n = client.read(&mut buf).unwrap_or(0);
    let reply = String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("").to_string();
    (reply, started.elapsed().as_millis())
}

fn seen(rx: &Receiver<String>) -> Vec<String> {
    let mut lines = Vec::new();
    while let Ok(line) = rx.recv_timeout(Duration::from_secs(3)) {
        lines.push(line);
    }
    lines
}

#[test]
fn page_proxy_follows_system_proxy() {
    let mode = std::env::var("PROBE_MODE").expect("PROBE_MODE");
    let port: u16 = std::env::var("PROBE_PROXY_PORT").map_or(18888, |p| p.parse().unwrap());
    let rx = fake_system_proxy(port);
    let page_proxy = crate::browser_proxy::address().expect("page proxy");

    let connect = "CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n";
    let (reply, ms) = ask(page_proxy, connect);
    let got = seen(&rx);
    println!("PROBE mode={mode} CONNECT example.com:443 -> reply={reply:?} {ms}ms proxy_saw={got:?}");
    let mut timings = vec![ms];
    for _ in 0..4 {
        let (_, ms) = ask(page_proxy, connect);
        timings.push(ms);
        seen(&rx);
    }
    println!("PROBE mode={mode} CONNECT timings_ms={timings:?}");

    let get = "GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n";
    let (get_reply, _) = ask(page_proxy, get);
    let get_seen = seen(&rx);
    println!("PROBE mode={mode} GET http://example.com/ -> reply={get_reply:?} proxy_saw={get_seen:?}");

    let private = "CONNECT 10.0.0.1.sslip.io:443 HTTP/1.1\r\nHost: 10.0.0.1.sslip.io:443\r\n\r\n";
    let (private_reply, _) = ask(page_proxy, private);
    let private_seen = seen(&rx);
    println!("PROBE mode={mode} CONNECT 10.0.0.1.sslip.io:443 -> reply={private_reply:?} proxy_saw={private_seen:?}");
    assert!(private_reply.contains("403"), "private name not refused: {private_reply:?}");
    assert!(private_seen.is_empty(), "private name reached the proxy: {private_seen:?}");

    match mode.as_str() {
        "none" => {
            assert!(reply.contains(" 200"), "direct CONNECT failed: {reply:?}");
            assert!(got.is_empty(), "no proxy is set, yet the proxy saw {got:?}");
        }
        "proxied" | "pac" => {
            assert_eq!(
                got,
                vec!["CONNECT example.com:443 HTTP/1.1".to_string()],
                "the system proxy did not get the page's CONNECT by name"
            );
            assert!(reply.contains(" 200"), "CONNECT via proxy failed: {reply:?}");
            assert_eq!(get_seen, vec!["GET http://example.com/ HTTP/1.1".to_string()]);
            if mode == "pac" {
                let other = "CONNECT example.org:443 HTTP/1.1\r\nHost: example.org:443\r\n\r\n";
                let (other_reply, _) = ask(page_proxy, other);
                let other_seen = seen(&rx);
                println!("PROBE mode=pac CONNECT example.org:443 (PAC says DIRECT) -> reply={other_reply:?} proxy_saw={other_seen:?}");
                assert!(other_seen.is_empty(), "PAC DIRECT host went to the proxy: {other_seen:?}");
                assert!(other_reply.contains(" 200"), "PAC DIRECT host failed: {other_reply:?}");
            }
        }
        other => panic!("unknown PROBE_MODE {other}"),
    }
    println!("PROBE PASS mode={mode}");
}

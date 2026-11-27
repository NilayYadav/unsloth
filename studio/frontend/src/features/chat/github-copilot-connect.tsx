// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { openExternalLink } from "@/lib/open-link";
import {
	disconnectGitHubCopilot,
	pollGitHubCopilot,
	startGitHubCopilot,
	type GitHubCopilotFlow,
} from "./api/providers-api";

export function GitHubCopilotConnect({
	providerId,
	connected,
	ensureProvider,
	onChanged,
}: {
	providerId: string | null;
	connected: boolean;
	ensureProvider: () => Promise<string>;
	onChanged: (id: string, connected: boolean) => Promise<void>;
}) {
	const [flow, setFlow] = useState<
		(GitHubCopilotFlow & { providerId: string }) | null
	>(null);
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");
	const generation = useRef(0);
	const previousProviderId = useRef(providerId);
	useEffect(() => {
		if (
			previousProviderId.current &&
			previousProviderId.current !== providerId
		) {
			generation.current += 1;
			setFlow(null);
			setBusy(false);
			setError("");
		}
		previousProviderId.current = providerId;
	}, [providerId]);
	const changed = useRef(onChanged);
	changed.current = onChanged;
	useEffect(
		() => () => {
			generation.current += 1;
		},
		[],
	);
	useEffect(() => {
		if (!flow) return;
		let cancelled = false;
		let timer: ReturnType<typeof setTimeout>;
		const expires = Date.now() + flow.expires_in * 1000;
		async function poll() {
			try {
				if (Date.now() >= expires)
					throw new Error("GitHub sign-in expired. Try again.");
				const result = await pollGitHubCopilot(flow!.providerId, flow!.flow_id);
				if (cancelled) return;
				if (result.status === "pending") {
					timer = setTimeout(
						poll,
						Math.max(5, result.interval ?? flow!.interval) * 1000,
					);
					return;
				}
				setFlow(null);
				if (result.status !== "connected")
					throw new Error(`GitHub sign-in ${result.status}. Try again.`);
				await changed.current(flow!.providerId, true);
			} catch (err) {
				if (!cancelled) {
					setFlow(null);
					setError(
						err instanceof Error ? err.message : "Could not sign in to GitHub.",
					);
				}
			}
		}
		timer = setTimeout(poll, flow.interval * 1000);
		return () => {
			cancelled = true;
			clearTimeout(timer);
		};
	}, [flow]);

	async function start() {
		const request = ++generation.current;
		setBusy(true);
		setError("");
		try {
			const id = providerId ?? (await ensureProvider());
			if (request !== generation.current) return;
			const next = await startGitHubCopilot(id);
			if (request === generation.current) setFlow({ ...next, providerId: id });
		} catch (err) {
			if (request === generation.current)
				setError(
					err instanceof Error
						? err.message
						: "Could not start GitHub sign-in.",
				);
		} finally {
			if (request === generation.current) setBusy(false);
		}
	}

	async function disconnect() {
		if (!providerId) return;
		++generation.current;
		setFlow(null);
		setBusy(true);
		setError("");
		try {
			await disconnectGitHubCopilot(providerId);
			await changed.current(providerId, false);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Could not disconnect GitHub.",
			);
		} finally {
			setBusy(false);
		}
	}

	return (
		<section className="space-y-3 rounded-xl border border-border/70 p-4">
			<p className="text-sm">
				Use your own GitHub Copilot subscription. Plan limits and organization
				policies apply. Text chat is supported.
			</p>
			{flow ? (
				<div className="space-y-2">
					<p className="text-sm">
						Enter this code on GitHub:{" "}
						<strong className="select-all font-mono">{flow.user_code}</strong>
					</p>
					<Button
						variant="outline"
						onClick={() =>
							void openExternalLink("https://github.com/login/device")
						}
					>
						Open GitHub sign-in
					</Button>
					<p className="text-xs text-muted-foreground">
						Waiting for authorization…
					</p>
				</div>
			) : (
				<Button variant="outline" disabled={busy} onClick={() => void start()}>
					{busy
						? "Connecting…"
						: connected
							? "Reconnect GitHub Copilot"
							: "Sign in with GitHub"}
				</Button>
			)}
			{(connected || flow) && (
				<Button
					variant="ghost"
					disabled={busy}
					onClick={() => void disconnect()}
				>
					Disconnect
				</Button>
			)}
			{error && (
				<p role="alert" className="text-sm text-destructive">
					{error}
				</p>
			)}
		</section>
	);
}

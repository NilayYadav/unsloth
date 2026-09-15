// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { loadCluster } from "../api/cluster";
import {
  CLUSTER_POLL_MS,
  CLUSTER_STARTING_POLL_MS,
  type ClusterState,
} from "../api/cluster-state";

export type ClusterRun = (
  key: string,
  request: () => Promise<ClusterState>,
) => Promise<string | null>;

export type ClusterController = {
  state: ClusterState | null;
  loadFailed: boolean;
  busy: ReadonlySet<string>;
  run: ClusterRun;
};

export function useClusterState(): ClusterController {
  const [state, setState] = useState<ClusterState | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [busy, setBusy] = useState<ReadonlySet<string>>(() => new Set());
  const [pollRevision, setPollRevision] = useState(0);
  const mutationEpoch = useRef(0);
  const inFlight = useRef(0);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRevision intentionally restarts polling after a mutation
  useEffect(() => {
    let stopped = false;
    let timer: number | null = null;
    const schedule = (delay: number) => {
      if (!stopped && inFlight.current === 0) {
        timer = window.setTimeout(poll, delay);
      }
    };
    const poll = () => {
      if (inFlight.current > 0) {
        return;
      }
      const epoch = mutationEpoch.current;
      loadCluster()
        .then((next) => {
          if (
            stopped ||
            inFlight.current > 0 ||
            mutationEpoch.current !== epoch
          ) {
            return;
          }
          setState(next);
          setLoadFailed(false);
          schedule(
            next.share.state === "starting"
              ? CLUSTER_STARTING_POLL_MS
              : CLUSTER_POLL_MS,
          );
        })
        .catch(() => {
          if (stopped || mutationEpoch.current !== epoch) {
            return;
          }
          setLoadFailed(true);
          schedule(CLUSTER_POLL_MS);
        });
    };
    poll();
    return () => {
      stopped = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [pollRevision]);

  const run = useCallback<ClusterRun>(async (key, request) => {
    mutationEpoch.current += 1;
    inFlight.current += 1;
    setBusy((current) => new Set(current).add(key));
    try {
      setState(await request());
      setLoadFailed(false);
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    } finally {
      inFlight.current -= 1;
      setBusy((current) => {
        const next = new Set(current);
        next.delete(key);
        return next;
      });
      // polling resumes once every mutation settles and reconciles the visible state
      if (inFlight.current === 0) {
        setPollRevision((revision) => revision + 1);
      }
    }
  }, []);

  return { state, loadFailed, busy, run };
}

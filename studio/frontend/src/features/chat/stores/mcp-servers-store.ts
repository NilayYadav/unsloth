// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

interface McpServersStore {
  revision: number;
  notifyServersChanged: () => void;
}

export const useMcpServersStore = create<McpServersStore>((set) => ({
  revision: 0,
  notifyServersChanged: () => set((s) => ({ revision: s.revision + 1 })),
}));

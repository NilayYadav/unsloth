// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  ChatMcpServersDialog,
  type McpServerConfig,
  createMcpServer,
  deleteMcpServer,
  listMcpServers,
  refreshMcpServerTools,
  useMcpServersStore,
} from "@/features/chat";
import { type TranslationKey, useT } from "@/i18n";
import { CheckmarkBadge01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { SettingsSection } from "../components/settings-section";

interface ConnectorDef {
  name: string;
  logo: string;
  descriptionKey: TranslationKey;
  url: string;
}

function connectorLogoSrc(file: string): string {
  return `${import.meta.env.BASE_URL}connector-logos/${file}`;
}

const MCP_ENDPOINTS = {
  gmail: "https://gmailmcp.googleapis.com/mcp/v1",
  drive: "https://drivemcp.googleapis.com/mcp/v1",
  calendar: "https://calendarmcp.googleapis.com/mcp/v1",
  linear: "https://mcp.linear.app/sse",
} as const;

const CONNECTORS: ConnectorDef[] = [
  {
    name: "Gmail",
    logo: "gmail.svg",
    descriptionKey: "settings.mcpConnectors.connectors.gmail",
    url: MCP_ENDPOINTS.gmail,
  },
  {
    name: "Drive",
    logo: "drive.svg",
    descriptionKey: "settings.mcpConnectors.connectors.drive",
    url: MCP_ENDPOINTS.drive,
  },
  {
    name: "Calendar",
    logo: "calendar.svg",
    descriptionKey: "settings.mcpConnectors.connectors.calendar",
    url: MCP_ENDPOINTS.calendar,
  },
  {
    name: "Linear",
    logo: "linear.svg",
    descriptionKey: "settings.mcpConnectors.connectors.linear",
    url: MCP_ENDPOINTS.linear,
  },
];

const TRAILING_SLASHES = /\/+$/;

function normalizeUrl(url: string): string {
  return url.trim().toLowerCase().replace(TRAILING_SLASHES, "");
}

export function McpConnectorsTab() {
  const t = useT();
  const [servers, setServers] = useState<McpServerConfig[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [busyUrl, setBusyUrl] = useState<string | null>(null);
  const revision = useMcpServersStore((s) => s.revision);
  const notifyServersChanged = useMcpServersStore(
    (s) => s.notifyServersChanged,
  );

  useEffect(() => {
    let cancelled = false;
    listMcpServers()
      .then((rows) => {
        if (!cancelled) setServers(rows);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [revision]);

  const serverForUrl = useCallback(
    (url: string) =>
      servers.find((s) => normalizeUrl(s.url) === normalizeUrl(url)),
    [servers],
  );

  async function connect(connector: ConnectorDef) {
    setBusyUrl(connector.url);
    let createdId: string | null = null;
    try {
      const existing = serverForUrl(connector.url);
      let serverId = existing?.id;
      if (!serverId) {
        const created = await createMcpServer({
          displayName: connector.name,
          url: connector.url,
          useOauth: true,
        });
        serverId = created.id;
        createdId = created.id;
      }
      const result = await refreshMcpServerTools(serverId);
      if (!result.ok) {
        throw new Error(result.error ?? t("settings.mcpConnectors.authFailed"));
      }
      toast.success(
        t("settings.mcpConnectors.connected", { name: connector.name }),
        {
          description: t("settings.mcpConnectors.toolsLoaded", {
            count: result.tool_count,
          }),
        },
      );
    } catch (err) {
      if (createdId) {
        await deleteMcpServer(createdId).catch(() => {});
      }
      toast.error(
        t("settings.mcpConnectors.connectError", { name: connector.name }),
        { description: err instanceof Error ? err.message : String(err) },
      );
    } finally {
      setBusyUrl(null);
      notifyServersChanged();
    }
  }

  async function disconnect(connector: ConnectorDef) {
    const server = serverForUrl(connector.url);
    if (!server) return;
    const ok = window.confirm(
      t("settings.mcpConnectors.disconnectConfirm", { name: connector.name }),
    );
    if (!ok) return;
    setBusyUrl(connector.url);
    try {
      await deleteMcpServer(server.id);
    } catch (err) {
      toast.error(
        t("settings.mcpConnectors.disconnectError", { name: connector.name }),
        { description: err instanceof Error ? err.message : String(err) },
      );
    } finally {
      setBusyUrl(null);
      notifyServersChanged();
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="font-heading text-lg font-semibold">
          {t("settings.mcpConnectors.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.mcpConnectors.description")}
        </p>
      </header>

      <SettingsSection
        title={t("settings.mcpConnectors.supportedTitle")}
        description={t("settings.mcpConnectors.supportedDescription")}
      >
        <div className="flex flex-col divide-y divide-border/60">
          {CONNECTORS.map((connector) => {
            const connected = Boolean(serverForUrl(connector.url));
            const available = connector.url !== "";
            const busy = busyUrl === connector.url;
            return (
              <div
                key={connector.name}
                className="flex items-center gap-3 py-3"
              >
                <div className="flex size-9 shrink-0 items-center justify-center rounded-[8px] border border-border bg-background">
                  <img
                    src={connectorLogoSrc(connector.logo)}
                    alt=""
                    aria-hidden={true}
                    className="size-5 object-contain"
                  />
                </div>
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="flex items-center gap-1.5 text-sm font-medium text-foreground">
                    {connector.name}
                    {connected ? (
                      <HugeiconsIcon
                        icon={CheckmarkBadge01Icon}
                        className="size-3.5 text-emerald-600 dark:text-emerald-400"
                      />
                    ) : null}
                  </span>
                  <span className="text-xs leading-snug text-muted-foreground">
                    {t(connector.descriptionKey)}
                  </span>
                </div>
                <div className="ml-auto flex shrink-0 items-center">
                  {available ? (
                    connected ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => disconnect(connector)}
                        disabled={busy}
                        className="text-muted-foreground hover:text-destructive"
                      >
                        {busy ? <Spinner /> : null}
                        {t("settings.mcpConnectors.disconnect")}
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        onClick={() => connect(connector)}
                        disabled={busy}
                      >
                        {busy ? <Spinner /> : null}
                        {busy
                          ? t("settings.mcpConnectors.connecting")
                          : t("settings.mcpConnectors.connect")}
                      </Button>
                    )
                  ) : (
                    <span className="text-xs font-medium text-muted-foreground">
                      {t("settings.mcpConnectors.comingSoon")}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.mcpConnectors.customTitle")}
        description={t("settings.mcpConnectors.customDescription")}
      >
        <div className="flex justify-start py-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setDialogOpen(true)}
          >
            <HugeiconsIcon icon={PlusSignIcon} size={14} className="mr-1.5" />
            {t("settings.mcpConnectors.customButton")}
          </Button>
        </div>
      </SettingsSection>

      <ChatMcpServersDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next);
          if (!next) notifyServersChanged();
        }}
      />
    </div>
  );
}

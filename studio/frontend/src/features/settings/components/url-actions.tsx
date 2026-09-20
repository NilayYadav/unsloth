// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { Copy01Icon, QrCodeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import QRCode from "react-qr-code";

export function CopyUrlButton({
  url,
  label = "Copy URL",
}: {
  url: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) {
        window.clearTimeout(copyTimer.current);
      }
    };
  }, []);
  const text = copied ? "Copied" : label;
  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="gap-1.5"
      aria-label={`${text} ${url}`}
      onClick={async () => {
        if (!(await copyToClipboard(url))) {
          return;
        }
        setCopied(true);
        if (copyTimer.current !== null) {
          window.clearTimeout(copyTimer.current);
        }
        copyTimer.current = window.setTimeout(() => setCopied(false), 1800);
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        className="size-3.5"
      />
      {text}
    </Button>
  );
}

export function UrlQrButton({
  url,
  description,
}: {
  url: string;
  description: string;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild={true}>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="gap-1.5"
          aria-label={`Show QR code for ${url}`}
        >
          <HugeiconsIcon icon={QrCodeIcon} className="size-3.5" />
          QR
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xs">
        <DialogHeader>
          <DialogTitle>Open on your phone</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <div className="mx-auto mt-2 rounded-md bg-white p-3">
          <QRCode value={url} size={192} />
        </div>
        <code className="block break-all text-center font-mono text-xs text-muted-foreground">
          {url}
        </code>
      </DialogContent>
    </Dialog>
  );
}

export function UrlActions({
  url,
  copyLabel,
  qrDescription,
}: {
  url: string;
  copyLabel?: string;
  qrDescription: string;
}) {
  return (
    <div className="flex shrink-0 items-center gap-2">
      <UrlQrButton url={url} description={qrDescription} />
      <CopyUrlButton url={url} label={copyLabel} />
    </div>
  );
}

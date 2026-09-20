// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";

/**
 * SettingsSection with the body behind a disclosure, for reference material
 * the primary flow does not need up front. The title stays mounted in the
 * trigger, so settings search still lands on a collapsed section.
 */
export function CollapsibleSettingsSection({
  title,
  description,
  children,
  defaultOpen = false,
}: {
  title: string;
  description?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <Collapsible defaultOpen={defaultOpen} className="flex flex-col">
      <CollapsibleTrigger
        data-collapsible-section={title}
        className="group flex w-full items-center justify-between gap-4 rounded-lg border border-border/70 bg-muted/30 px-4 py-3 text-left transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        <span className="flex min-w-0 flex-col gap-0.5">
          <span
            data-settings-label={title}
            className="text-sm font-medium text-foreground"
          >
            {title}
          </span>
          {description ? (
            <span className="text-xs text-muted-foreground leading-relaxed">
              {description}
            </span>
          ) : null}
        </span>
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-3">{children}</CollapsibleContent>
    </Collapsible>
  );
}

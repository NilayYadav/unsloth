// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { type DecisionMetrics, confusionTable } from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { DashboardSquare01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

function percent(value: number | null): string {
  return value === null ? "-" : `${(value * 100).toFixed(1)}%`;
}

function score(value: number | null): string {
  return value === null ? "-" : value.toFixed(3);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
    </div>
  );
}

function ConfusionTable({
  question,
}: {
  question: DecisionMetrics["questions"][number];
}) {
  const t = useT();
  const { options, rows } = confusionTable(question.confusion);
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-sm font-medium">{question.name}</h4>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead />
            {options.map((option) => (
              <TableHead key={option} className="text-right">
                {option}
              </TableHead>
            ))}
            <TableHead className="text-right">
              {t("studio.progress.heldoutPrecision")}
            </TableHead>
            <TableHead className="text-right">
              {t("studio.progress.heldoutRecall")}
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, i) => (
            <TableRow key={row.option}>
              <TableHead scope="row">{row.option}</TableHead>
              {row.counts.map((count, j) => (
                <TableCell
                  key={options[j]}
                  className={cn(
                    "text-right tabular-nums",
                    i === j && "font-semibold",
                    count === 0 && "text-muted-foreground",
                  )}
                >
                  {count}
                </TableCell>
              ))}
              <TableCell className="text-right tabular-nums">
                {percent(row.precision)}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {percent(row.recall)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function DecisionMetricsSection({
  metrics,
}: {
  metrics: DecisionMetrics;
}): ReactElement {
  const t = useT();
  return (
    <SectionCard
      icon={<HugeiconsIcon icon={DashboardSquare01Icon} className="size-5" />}
      title={t("studio.progress.heldoutTitle")}
      description={t("studio.progress.heldoutDescription", {
        count: metrics.decisions.toLocaleString(),
      })}
      className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap gap-8">
          <Stat
            label={t("studio.progress.heldoutBaseAccuracy")}
            value={percent(metrics.accuracy_base)}
          />
          <Stat
            label={t("studio.progress.heldoutTunedAccuracy")}
            value={percent(metrics.accuracy)}
          />
          <Stat
            label={t("studio.progress.heldoutCalibrationError")}
            value={`${score(metrics.ece_base)} → ${score(metrics.ece)}`}
          />
        </div>
        {metrics.questions.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {t("studio.progress.heldoutTableHint")}
          </p>
        )}
        {metrics.questions.map((question) => (
          <ConfusionTable key={question.name} question={question} />
        ))}
        {metrics.questions.length < metrics.questions_total && (
          <p className="text-xs text-muted-foreground">
            {t("studio.progress.heldoutQuestionsShown", {
              shown: metrics.questions.length,
              total: metrics.questions_total,
            })}
          </p>
        )}
      </div>
    </SectionCard>
  );
}

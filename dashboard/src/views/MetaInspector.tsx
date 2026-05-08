import { useEffect, useState } from "react";
import { fetchHealth } from "../lib/api";
import type { HealthResponse } from "../lib/api";

export interface MetaInspectorProps {
  [key: string]: unknown;
}

export function MetaInspector({}: MetaInspectorProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [fromMock, setFromMock] = useState(false);

  useEffect(() => {
    (async () => {
      const { data, fromMock } = await fetchHealth();
      setHealth(data);
      setFromMock(fromMock);
    })();
  }, []);

  if (!health) {
    return null;
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
      <StatBlock
        label="Version"
        value={health.version}
        accent={fromMock ? "var(--color-konjo-warm)" : "var(--color-konjo-fg)"}
      />
      <StatBlock
        label="Platform"
        value={health.platform}
        accent="var(--color-konjo-accent)"
      />
      <StatBlock
        label="Python"
        value={health.python_version}
        accent="var(--color-konjo-fg-muted)"
      />
      <StatBlock
        label="Status"
        value={fromMock ? "mock" : "live"}
        accent={fromMock ? "var(--color-konjo-warm)" : "var(--color-konjo-good)"}
      />
      <StatBlock
        label="Uptime"
        value={health.uptime_seconds > 0 ? `${Math.floor(health.uptime_seconds)}s` : "—"}
        accent="var(--color-konjo-fg-muted)"
      />
      <StatBlock
        label="Source"
        value={fromMock ? "offline" : "server"}
        accent={fromMock ? "var(--color-konjo-warm)" : "var(--color-konjo-good)"}
      />
    </div>
  );
}

function StatBlock({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2 rounded-konjo bg-konjo-surface/60 border border-konjo-line/60">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
        {label}
      </div>
      <div
        className="text-konjo-mono text-[12px] font-medium"
        style={{ color: accent ?? "var(--color-konjo-fg)" }}
      >
        {value}
      </div>
    </div>
  );
}

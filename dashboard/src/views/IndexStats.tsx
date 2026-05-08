import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { fetchIndexStats } from "../lib/api";
import type { IndexStats } from "../lib/api";

export function IndexStats() {
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [fromMock, setFromMock] = useState(false);

  useEffect(() => {
    (async () => {
      const { data, fromMock } = await fetchIndexStats();
      setStats(data);
      setFromMock(fromMock);
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return (
      <div className="glass-konjo rounded-konjo-lg p-5 h-32 flex items-center justify-center">
        <p className="text-konjo-fg-muted text-konjo-mono text-[12px]">loading corpus stats…</p>
      </div>
    );
  }

  if (!stats) {
    return null;
  }

  const bytesFormat = (bytes: number) => {
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Corpus Statistics
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Index snapshot · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: ease.kanjo }}
        className="glass-konjo rounded-konjo-lg p-5"
      >
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard
            label="Vectors"
            value={stats.corpus_size_vectors}
            unit="entries"
            color="var(--color-konjo-good)"
          />
          <StatCard
            label="Original (FP32)"
            value={bytesFormat(stats.corpus_bytes_fp32)}
            unit=""
            color="var(--color-konjo-fg)"
          />
          <StatCard
            label="Compressed"
            value={bytesFormat(stats.corpus_bytes_compressed)}
            unit=""
            color="var(--color-konjo-accent)"
          />
          <StatCard
            label="Compression Ratio"
            value={stats.compression_ratio.toFixed(2)}
            unit="×"
            color="var(--color-konjo-good)"
          />

          <StatCard
            label="HNSW Capacity"
            value={stats.hnsw_capacity}
            unit="vectors"
            color="var(--color-konjo-fg-muted)"
          />
          <StatCard
            label="HNSW Ef (construct)"
            value={stats.hnsw_ef_construction}
            unit=""
            color="var(--color-konjo-fg-muted)"
          />
          <div className="sm:col-span-2 lg:col-span-2 bg-konjo-surface/60 rounded p-3">
            <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted mb-2">
              Compression Savings
            </div>
            <div className="text-[24px] font-bold text-konjo-good">
              {(
                ((stats.corpus_bytes_fp32 - stats.corpus_bytes_compressed) / stats.corpus_bytes_fp32) *
                100
              ).toFixed(0)}
              %
            </div>
            <div className="text-konjo-mono text-[10px] text-konjo-fg-muted mt-1">
              {bytesFormat(stats.corpus_bytes_fp32 - stats.corpus_bytes_compressed)} saved
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}

function StatCard({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: string | number;
  unit: string;
  color: string;
}) {
  return (
    <div className="bg-konjo-surface/60 rounded p-3">
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted mb-2">
        {label}
      </div>
      <div className="flex items-baseline gap-1">
        <div className="text-[22px] font-bold tabular-nums" style={{ color }}>
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>
        {unit && <div className="text-konjo-fg-muted text-[11px]">{unit}</div>}
      </div>
    </div>
  );
}

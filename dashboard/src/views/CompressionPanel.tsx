import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import { compressVectors } from "../lib/api";
import type { CompressResult } from "../lib/api";

export function CompressionPanel() {
  const [batchSize, setBatchSize] = useState(1000);
  const [dimensions, setDimensions] = useState(768);
  const [mode, setMode] = useState<"int8" | "nf4" | "binary">("int8");
  const [result, setResult] = useState<CompressResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  useEffect(() => {
    setLoading(true);
    (async () => {
      const { data, fromMock } = await compressVectors(batchSize, dimensions, mode);
      setResult(data);
      setFromMock(fromMock);
      setLoading(false);
    })();
  }, [batchSize, dimensions, mode]);

  const modeColor: Record<typeof mode, string> = {
    int8: "var(--color-konjo-good)",
    nf4: "var(--color-konjo-accent)",
    binary: "var(--color-konjo-warm)",
  };

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Compression Forge
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Real Vectro.compress() · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div className="space-y-2">
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
            Compression Mode
          </label>
          <div className="flex gap-2">
            {(["int8", "nf4", "binary"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={[
                  "px-3 py-2 rounded-konjo text-[12px] font-mono uppercase tracking-[0.14em] transition-colors",
                  mode === m
                    ? "bg-konjo-accent text-konjo-bg"
                    : "border border-konjo-line text-konjo-fg-muted hover:text-konjo-fg",
                ].join(" ")}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="grid sm:grid-cols-2 gap-3">
          <div>
            <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-1 block">
              Batch Size
            </label>
            <input
              type="range"
              min="100"
              max="10000"
              step="100"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-konjo-fg text-[12px] mt-1">{batchSize.toLocaleString()} vectors</div>
          </div>

          <div>
            <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-1 block">
              Dimensions
            </label>
            <div className="flex gap-2">
              {[256, 768, 1536].map((d) => (
                <button
                  key={d}
                  onClick={() => setDimensions(d)}
                  className={[
                    "px-2 py-1 rounded-konjo text-[11px] font-mono transition-colors",
                    dimensions === d
                      ? "bg-konjo-accent text-konjo-bg"
                      : "border border-konjo-line text-konjo-fg-muted",
                  ].join(" ")}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>
        </div>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3, ease: ease.kanjo }}
              className="pt-4 border-t border-konjo-line/40 space-y-3"
            >
              <div className="grid sm:grid-cols-3 gap-3 text-center">
                <div className="bg-konjo-surface rounded p-3">
                  <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted mb-1">
                    Compression Ratio
                  </div>
                  <div className="text-[28px] font-bold" style={{ color: modeColor[mode] }}>
                    {result.compression_ratio.toFixed(2)}<span className="text-[16px] ml-1">×</span>
                  </div>
                </div>

                <div className="bg-konjo-surface rounded p-3">
                  <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted mb-1">
                    Cosine Similarity
                  </div>
                  <div className="text-[28px] font-bold text-konjo-good">
                    {result.cosine_similarity.toFixed(4)}
                  </div>
                </div>

                <div className="bg-konjo-surface rounded p-3">
                  <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted mb-1">
                    Throughput
                  </div>
                  <div className="text-[24px] font-bold text-konjo-accent">
                    {result.throughput_m_vec_s.toFixed(2)} <span className="text-[12px]">M vec/s</span>
                  </div>
                </div>
              </div>

              <div className="text-konjo-mono text-[11px] text-konjo-fg-muted space-y-1">
                <div>
                  original: {(result.original_bytes / 1024 / 1024).toFixed(1)} MB · compressed:{" "}
                  {(result.compressed_bytes / 1024).toFixed(0)} KB
                </div>
                <div>elapsed: {result.elapsed_ms.toFixed(1)} ms</div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {loading && (
          <div className="text-konjo-mono text-[11px] text-konjo-fg-muted animate-pulse">
            compressing…
          </div>
        )}
      </div>
    </section>
  );
}

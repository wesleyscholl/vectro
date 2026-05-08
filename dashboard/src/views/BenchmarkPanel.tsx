import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Dial, ease } from "@konjoai/ui";
import { benchmarkInt8 } from "../lib/api";
import type { BenchmarkResult } from "../lib/api";

export function BenchmarkPanel() {
  const [dimensions, setDimensions] = useState(768);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  useEffect(() => {
    setLoading(true);
    (async () => {
      const { data, fromMock } = await benchmarkInt8(dimensions);
      setResult(data);
      setFromMock(fromMock);
      setLoading(false);
    })();
  }, [dimensions]);

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            INT8 Benchmark
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Real NEON/AVX2 throughput · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div>
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2 block">
            Dimensions
          </label>
          <div className="flex gap-2">
            {[256, 512, 768, 1536].map((d) => (
              <button
                key={d}
                onClick={() => setDimensions(d)}
                className={[
                  "px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
                  dimensions === d
                    ? "bg-konjo-accent text-konjo-bg"
                    : "border border-konjo-line text-konjo-fg-muted hover:text-konjo-fg",
                ].join(" ")}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3, ease: ease.kanjo }}
              className="space-y-4"
            >
              <div className="grid sm:grid-cols-[auto_1fr] gap-5 items-center">
                <motion.div
                  key={`${dimensions}-${result.throughput_m_vec_s}`}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, ease: ease.kanjo }}
                >
                  <Dial
                    value={result.throughput_m_vec_s}
                    min={0}
                    max={20}
                    unit="M vec/s"
                    label="INT8 Throughput"
                    severity={result.throughput_m_vec_s > 10 ? "ok" : "warn"}
                    format={(v) => v.toFixed(2)}
                    size={150}
                    sublabel={loading ? "measuring…" : "live"}
                  />
                </motion.div>

                <div className="space-y-2">
                  <div className="space-y-1">
                    <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
                      Benchmark Summary
                    </div>
                    <div className="space-y-2 text-[12px]">
                      <div className="flex justify-between">
                        <span className="text-konjo-fg-muted">Total vectors:</span>
                        <span className="text-konjo-fg font-mono">
                          {result.total_vectors.toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-konjo-fg-muted">Dimensions:</span>
                        <span className="text-konjo-fg font-mono">{result.dimensions}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-konjo-fg-muted">Elapsed:</span>
                        <span className="text-konjo-fg font-mono">{result.elapsed_ms.toFixed(1)} ms</span>
                      </div>
                      <div className="flex justify-between font-semibold pt-2 border-t border-konjo-line/40">
                        <span className="text-konjo-accent">Throughput:</span>
                        <span className="text-konjo-accent font-mono">
                          {result.throughput_m_vec_s.toFixed(2)} M vec/s
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {loading && (
          <div className="text-konjo-mono text-[11px] text-konjo-fg-muted animate-pulse">
            benchmarking…
          </div>
        )}
      </div>
    </section>
  );
}

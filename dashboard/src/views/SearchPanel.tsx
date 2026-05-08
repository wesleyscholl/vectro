import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import { searchVectors } from "../lib/api";
import type { SearchResult } from "../lib/api";

const SAMPLE_QUERIES = [
  "What is machine learning?",
  "How does neural networks work?",
  "Explain embeddings and similarity",
];

export function SearchPanel() {
  const [query, setQuery] = useState("How does semantic search work?");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    const { data, fromMock } = await searchVectors(query, 5);
    setResult(data);
    setFromMock(fromMock);
    setLoading(false);
  };

  return (
    <section className="space-y-3">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Semantic Search
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Real DSPy retriever · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
          </p>
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div className="space-y-2">
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
            Query
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="w-full px-3 py-2 bg-konjo-surface border border-konjo-line rounded-konjo text-konjo-fg text-[13px] placeholder-konjo-fg-muted resize-none h-16"
          />
        </div>

        <div className="space-y-2">
          <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
            Sample queries
          </div>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                onClick={() => setQuery(q)}
                className="px-2 py-1 text-[11px] bg-konjo-surface border border-konjo-line rounded-konjo text-konjo-fg-muted hover:text-konjo-fg transition-colors"
              >
                {q.substring(0, 20)}…
              </button>
            ))}
          </div>
        </div>

        <motion.button
          onClick={handleSearch}
          disabled={loading}
          whileHover={!loading ? { scale: 1.02 } : undefined}
          whileTap={!loading ? { scale: 0.98 } : undefined}
          className="w-full px-4 py-2 bg-konjo-accent text-konjo-bg rounded-konjo text-[13px] font-medium uppercase tracking-[0.16em] disabled:opacity-50"
        >
          {loading ? "Searching…" : "Search"}
        </motion.button>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3, ease: ease.kanjo }}
              className="pt-4 border-t border-konjo-line/40 space-y-2"
            >
              <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted mb-3">
                Results (top 5)
              </div>
              {result.passages.map((p, i) => (
                <div key={i} className="bg-konjo-surface rounded p-2">
                  <div className="flex justify-between items-start gap-2 mb-1">
                    <div className="text-konjo-mono text-[10px] text-konjo-accent">
                      #{p.rank} · {(p.score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <p className="text-konjo-fg text-[12px] line-clamp-2">{p.text}</p>
                </div>
              ))}
              <div className="text-konjo-mono text-[10px] text-konjo-fg-muted pt-2 border-t border-konjo-line/40">
                {result.elapsed_ms.toFixed(1)} ms
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}

import { KonjoApp } from "@konjoai/ui";
import { CompressionPanel } from "./views/CompressionPanel";
import { SearchPanel } from "./views/SearchPanel";
import { BenchmarkPanel } from "./views/BenchmarkPanel";
import { IndexStats } from "./views/IndexStats";
import { MetaInspector } from "./views/MetaInspector";

export default function App() {
  return (
    <KonjoApp
      product="vectro"
      tagline="Quantization Forge · ultra-high-performance embedding compression"
      status={{ label: "ready", severity: "ok" }}
    >
      <Hero />

      <div className="space-y-6 mt-10">
        <IndexStats />

        <section className="grid lg:grid-cols-2 gap-4">
          <CompressionPanel />
          <SearchPanel />
        </section>

        <BenchmarkPanel />

        <MetaInspector />

        <Footer />
      </div>
    </KonjoApp>
  );
}

function Hero() {
  return (
    <section className="text-center pt-6 pb-2">
      <p className="text-konjo-mono uppercase tracking-[0.32em] text-konjo-violet" style={{ fontSize: 11 }}>
        vectro · 向量 · vector · ベクトル
      </p>
      <h1
        className="text-konjo-display text-konjo-fg mt-4 mx-auto"
        style={{ fontSize: 52, fontWeight: 600, letterSpacing: "-0.025em", maxWidth: 920, lineHeight: 1.05 }}
      >
        Embeddings, <span style={{ color: "var(--color-konjo-accent)" }}>compressed</span>.
      </h1>
      <p
        className="text-konjo-fg-muted mt-5 mx-auto"
        style={{ fontSize: 16, maxWidth: 640, lineHeight: 1.55 }}
      >
        INT8 · NF4 · Binary · PQ. Every compression mode, measured in real-time. Lossy, lossless, and everything in between.
        See the tradeoffs, then ship.
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer
      className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-fg-muted text-konjo-mono"
      style={{ fontSize: 12 }}
    >
      <div className="flex flex-wrap gap-4 justify-between items-baseline">
        <span>
          built on{" "}
          <span className="text-konjo-fg">@konjoai/ui</span>
          {" · "}
          <span className="text-konjo-fg">/api/compress</span>
          {" · "}
          <span className="text-konjo-fg">/api/search</span>
          {" · "}
          <span className="text-konjo-fg">/api/benchmark</span>
          {" · "}
          <span className="text-konjo-fg">/api/index-stats</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio · squish · kyro · miru · kohaku · kairu · toki · squash
        </span>
      </div>
    </footer>
  );
}

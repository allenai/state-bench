"use client";

import { useEffect, useMemo, useState } from "react";

type Complexity = "periodic" | "aperiodic" | "r-trivial";
type PaddingComplexity = "none" | "easy" | "medium" | "hard";
type Operation =
  | { op: "constant"; target: number; value: number }
  | { op: "variable"; target: number; source: number };

type Sample = {
  id: string;
  sampleIndex: number;
  sourceInstanceId: string;
  formatter: string;
  complexity: Complexity;
  paddingComplexity: PaddingComplexity;
  paddingPercent: number;
  sourceOperations: Operation[];
  operations: Operation[];
  query: number[];
  finalState: number[];
  prefix: string;
  completion: string;
  numFillerLines: number;
  metadata: Record<string, unknown>;
};

type SamplePack = {
  schemaVersion: number;
  samplesPerCondition: number;
  samples: Sample[];
};

const COMPLEXITIES: Complexity[] = ["periodic", "aperiodic", "r-trivial"];
const PADDING_PERCENTAGES = [0, 25, 50, 75, 90];
const PADDING_STYLES: PaddingComplexity[] = ["easy", "medium", "hard"];

const COMPLEXITY_LABELS: Record<Complexity, string> = {
  periodic: "Periodic",
  aperiodic: "Aperiodic",
  "r-trivial": "R-trivial",
};

export function Inspector() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [formatter, setFormatter] = useState("integer-code");
  const [complexity, setComplexity] = useState<Complexity>("periodic");
  const [paddingPercent, setPaddingPercent] = useState(0);
  const [paddingComplexity, setPaddingComplexity] =
    useState<PaddingComplexity>("none");
  const [sampleIndex, setSampleIndex] = useState(1);
  const [showMetadata, setShowMetadata] = useState(false);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    fetch("/samples.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error("Sample pack unavailable");
        return response.json() as Promise<SamplePack>;
      })
      .then((pack) => setSamples(pack.samples))
      .catch(() => setLoadError(true));
  }, []);

  const formatters = useMemo(
    () => [...new Set(samples.map((sample) => sample.formatter))].sort(),
    [samples],
  );
  const sample = samples.find(
    (candidate) =>
      candidate.formatter === formatter &&
      candidate.complexity === complexity &&
      candidate.paddingPercent === paddingPercent &&
      candidate.paddingComplexity === paddingComplexity &&
      candidate.sampleIndex === sampleIndex,
  );

  function changePaddingPercent(value: number) {
    setPaddingPercent(value);
    setPaddingComplexity(value === 0 ? "none" : "easy");
  }

  return (
    <main className="app-shell">
      <section className="controls" aria-label="Sequence filters">
        <label>
          <span>Formatting</span>
          <select value={formatter} onChange={(event) => setFormatter(event.target.value)}>
            {formatters.length ? (
              formatters.map((name) => (
                <option key={name} value={name}>
                  {formatLabel(name)}
                </option>
              ))
            ) : (
              <option value="integer-code">Loading…</option>
            )}
          </select>
        </label>
        <label>
          <span>Monoid complexity</span>
          <select
            value={complexity}
            onChange={(event) => setComplexity(event.target.value as Complexity)}
          >
            {COMPLEXITIES.map((value) => (
              <option key={value} value={value}>
                {COMPLEXITY_LABELS[value]}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Padding amount</span>
          <select
            value={paddingPercent}
            onChange={(event) => changePaddingPercent(Number(event.target.value))}
          >
            {PADDING_PERCENTAGES.map((value) => (
              <option key={value} value={value}>
                {value}%
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Padding complexity</span>
          <select
            value={paddingComplexity}
            disabled={paddingPercent === 0}
            onChange={(event) =>
              setPaddingComplexity(event.target.value as PaddingComplexity)
            }
          >
            {paddingPercent === 0 ? (
              <option value="none">None</option>
            ) : (
              PADDING_STYLES.map((value) => (
                <option key={value} value={value}>
                  {formatLabel(value)}
                </option>
              ))
            )}
          </select>
        </label>
      </section>

      {loadError ? (
        <section className="empty-state">The bundled sample pack could not be loaded.</section>
      ) : sample ? (
        <>
          <section className="sample-toolbar">
            <div className="sample-meta">
              <span className={`complexity-dot ${complexity}`} aria-hidden="true" />
              <strong>{COMPLEXITY_LABELS[complexity]}</strong>
              <span>{sample.operations.length} assignments</span>
              <span>{sample.numFillerLines} padding lines</span>
              <code>{sample.sourceInstanceId.slice(0, 10)}</code>
            </div>
            <div className="sample-actions">
              <div className="sample-switcher" aria-label="Choose sample">
                {[1, 2].map((value) => (
                  <button
                    className={sampleIndex === value ? "active" : ""}
                    key={value}
                    onClick={() => setSampleIndex(value)}
                    type="button"
                  >
                    Sample {value}
                  </button>
                ))}
              </div>
              <button
                aria-expanded={showMetadata}
                className={`metadata-toggle ${showMetadata ? "active" : ""}`}
                onClick={() => setShowMetadata((visible) => !visible)}
                type="button"
              >
                {showMetadata ? "Hide metadata" : "Show metadata"}
              </button>
            </div>
          </section>

          {showMetadata && (
            <section className="metadata-panel" aria-label="Sample metadata">
              <pre>{JSON.stringify(sample.metadata, null, 2)}</pre>
            </section>
          )}

          <section className="workspace">
            <article className="trace-panel">
              <div className="panel-heading">
                <h2>Source → derived trace</h2>
                <span className="rewrite-count">
                  {rewriteCount(sample)} rewritten
                </span>
              </div>
              <div className="trace-columns" aria-hidden="true">
                <span>Periodic source</span>
                <span>{COMPLEXITY_LABELS[complexity]} output</span>
              </div>
              <ol className="operations">
                {sample.sourceOperations.map((sourceOperation, index) => {
                  const operation = sample.operations[index];
                  const rewritten = !sameOperation(sourceOperation, operation);
                  return (
                    <li className={rewritten ? "rewritten" : ""} key={`${sample.id}-${index}`}>
                      <span className="line-number">{String(index + 1).padStart(2, "0")}</span>
                      <code>{operationLabel(sourceOperation)}</code>
                      <span className="operation-arrow">→</span>
                      <code>{operationLabel(operation)}</code>
                      {rewritten && <span className="rewrite-tag">resolved</span>}
                    </li>
                  );
                })}
              </ol>
              <div className="state-readout">
                <span>Final state</span>
                {sample.query.map((variable) => (
                  <code key={variable}>
                    v{variable} = {sample.finalState[variable]}
                  </code>
                ))}
              </div>
            </article>

            <article className="render-panel">
              <div className="panel-heading">
                <h2>{formatLabel(sample.formatter)}</h2>
                <span className="padding-badge">
                  {sample.paddingPercent}% / {formatLabel(sample.paddingComplexity)}
                </span>
              </div>
              <div className="prompt-window">
                <div className="window-bar">
                  <span>prompt</span>
                  <span>{sample.prefix.split("\n").length} lines</span>
                </div>
                <pre>{sample.prefix}</pre>
              </div>
              <div className="completion-block">
                <span>Expected completion</span>
                <pre>{sample.completion}</pre>
              </div>
            </article>
          </section>
        </>
      ) : (
        <section className="empty-state">Loading generated samples…</section>
      )}
    </main>
  );
}

function operationLabel(operation: Operation): string {
  return operation.op === "constant"
    ? `v${operation.target} ← ${operation.value}`
    : `v${operation.target} ← v${operation.source}`;
}

function sameOperation(left: Operation, right: Operation): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function rewriteCount(sample: Sample): number {
  return sample.operations.filter(
    (operation, index) => !sameOperation(operation, sample.sourceOperations[index]),
  ).length;
}

function formatLabel(value: string): string {
  return value
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

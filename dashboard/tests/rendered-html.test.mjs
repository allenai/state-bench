import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the sequence inspector", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>State Bench Sequence Inspector<\/title>/i);
  assert.match(html, /aria-label="Sequence filters"/);
  assert.match(html, /Formatting/);
  assert.match(html, /Monoid complexity/);
  assert.match(html, /Loading generated samples/);
  assert.doesNotMatch(
    html,
    /Inspect every derivation|source of truth|codex-preview|react-loading-skeleton/,
  );
});

test("bundles complete deterministic inspection conditions", async () => {
  const payload = JSON.parse(
    await readFile(new URL("../public/samples.json", import.meta.url), "utf8"),
  );
  assert.equal(payload.samplesPerCondition, 2);
  assert.equal(payload.samples.length, 468);
  assert.deepEqual(
    [...new Set(payload.samples.map((sample) => sample.complexity))].sort(),
    ["aperiodic", "periodic", "r-trivial"],
  );
  assert.equal(
    new Set(
      payload.samples.map(
        (sample) =>
          `${sample.formatter}/${sample.complexity}/${sample.paddingComplexity}/${sample.paddingPercent}`,
      ),
    ).size,
    234,
  );
  assert.ok(payload.samples.every((sample) => sample.metadata));
  assert.ok(payload.samples.some((sample) => sample.formatter === "ruler"));
  assert.ok(
    payload.samples.every(
      (sample) => sample.metadata.incorrect_completions.length === 4,
    ),
  );
  assert.ok(
    payload.samples.every((sample) => sample.metadata.choices.length === 5),
  );
  assert.ok(
    payload.samples.every((sample) => {
      const index = sample.metadata.correct_choice_index;
      return sample.metadata.choices[index] === sample.completion;
    }),
  );
});

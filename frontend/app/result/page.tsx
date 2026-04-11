/**
 * app/result/page.tsx
 * Result page — displays the document summary and generated questions,
 * and renders the Feedback component for thumbs-up/thumbs-down rating.
 */

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Feedback from "@/components/Feedback";

interface InferenceResult {
  doc_id: string;
  variant: string;
  summary: string | null;
  questions: string[] | null;
  latency_ms: number;
}

export default function ResultPage() {
  const router = useRouter();
  const [result, setResult] = useState<InferenceResult | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("ai_doc_result");
    if (!stored) {
      // No result in storage — redirect to upload page
      router.replace("/");
      return;
    }
    try {
      setResult(JSON.parse(stored) as InferenceResult);
    } catch {
      router.replace("/");
    }
  }, [router]);

  if (!result) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="text-slate-400 text-sm">Loading result…</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Analysis Result</h1>
          <p className="text-sm text-slate-400 mt-1">
            Variant&nbsp;
            <span className="font-mono font-semibold text-brand-600">{result.variant}</span>
            &nbsp;&mdash;&nbsp;processed in&nbsp;
            <span className="font-mono">{result.latency_ms.toFixed(0)}&nbsp;ms</span>
          </p>
        </div>
        <button onClick={() => router.push("/")} className="btn-ghost text-sm">
          Upload another
        </button>
      </div>

      {/* Summary section */}
      {result.summary && (
        <section className="card">
          <h2 className="text-lg font-semibold text-slate-800 mb-3">Summary</h2>
          <p className="text-slate-600 leading-relaxed text-sm whitespace-pre-wrap">
            {result.summary}
          </p>
        </section>
      )}

      {/* Generated questions section */}
      {result.questions && result.questions.length > 0 && (
        <section className="card">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">
            Generated Questions
            <span className="ml-2 text-xs font-normal text-slate-400">
              ({result.questions.length})
            </span>
          </h2>
          <ol className="flex flex-col gap-3">
            {result.questions.map((question, idx) => (
              <li key={idx} className="flex gap-3 text-sm text-slate-700">
                <span className="flex-shrink-0 rounded-full bg-brand-50 text-brand-600 font-semibold
                                  h-6 w-6 flex items-center justify-center text-xs">
                  {idx + 1}
                </span>
                <span className="leading-relaxed pt-0.5">{question}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {/* No inference results fallback */}
      {!result.summary && (!result.questions || result.questions.length === 0) && (
        <div className="card text-center text-slate-400 py-10">
          No summary or questions were generated for this document.
        </div>
      )}

      {/* Feedback component */}
      <Feedback docId={result.doc_id} variant={result.variant} />
    </div>
  );
}

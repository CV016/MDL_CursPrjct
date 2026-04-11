/**
 * components/Feedback.tsx
 * Thumbs-up / thumbs-down feedback widget.
 *
 * Props:
 *   docId   - The UUID of the document that was analysed.
 *   variant - The A/B variant used for the inference ("A" or "B").
 *
 * On button click, the component POSTs { doc_id, variant, score } to the
 * gateway POST /feedback endpoint and shows a confirmation message.
 */

"use client";

import { useState } from "react";
import { apiPost, ApiError } from "@/services/api";

interface FeedbackProps {
  docId: string;
  variant: string;
}

type FeedbackState = "idle" | "submitting" | "submitted" | "error";

export default function Feedback({ docId, variant }: FeedbackProps) {
  const [state, setState] = useState<FeedbackState>("idle");
  const [selectedScore, setSelectedScore] = useState<1 | -1 | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const handleFeedback = async (score: 1 | -1) => {
    if (state === "submitted" || state === "submitting") return;

    setSelectedScore(score);
    setState("submitting");
    setErrorMessage("");

    try {
      await apiPost("/feedback", {
        doc_id: docId,
        variant,
        score,
      });
      setState("submitted");
    } catch (err: unknown) {
      const message =
        err instanceof ApiError
          ? err.detail
          : "Could not submit feedback. Please try again.";
      setErrorMessage(message);
      setState("error");
      setSelectedScore(null);
    }
  };

  /* -------------------------------------------------------------------------
   * Render
   * ---------------------------------------------------------------------- */

  if (state === "submitted") {
    return (
      <div className="card flex items-center gap-3 text-sm text-slate-600">
        {/* Checkmark icon */}
        <span className="flex-shrink-0 rounded-full bg-green-100 p-1.5">
          <svg className="h-4 w-4 text-green-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M16.704 5.296a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-3-3a1 1 0 111.414-1.414L9 11.586l6.29-6.29a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </span>
        Thank you for your feedback — it helps improve the model.
      </div>
    );
  }

  return (
    <div className="card">
      <p className="text-sm font-medium text-slate-700 mb-4">
        Was this analysis helpful?
      </p>

      <div className="flex gap-3">
        {/* Thumbs up */}
        <button
          onClick={() => handleFeedback(1)}
          disabled={state === "submitting"}
          aria-label="Thumbs up — helpful"
          className={[
            "flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors",
            selectedScore === 1
              ? "border-green-400 bg-green-50 text-green-700"
              : "border-slate-200 bg-white text-slate-600 hover:border-green-300 hover:bg-green-50 hover:text-green-700",
            state === "submitting" ? "cursor-not-allowed opacity-50" : "cursor-pointer",
          ].join(" ")}
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21H5a2 2 0 01-2-2v-7a2 2 0 012-2h2.5M14 10V5a2 2 0 00-2-2H9l-3 5v8"
            />
          </svg>
          Helpful
        </button>

        {/* Thumbs down */}
        <button
          onClick={() => handleFeedback(-1)}
          disabled={state === "submitting"}
          aria-label="Thumbs down — not helpful"
          className={[
            "flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors",
            selectedScore === -1
              ? "border-red-400 bg-red-50 text-red-700"
              : "border-slate-200 bg-white text-slate-600 hover:border-red-300 hover:bg-red-50 hover:text-red-700",
            state === "submitting" ? "cursor-not-allowed opacity-50" : "cursor-pointer",
          ].join(" ")}
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3H19a2 2 0 012 2v7a2 2 0 01-2 2h-2.5M10 14v5a2 2 0 002 2h3l3-5v-8"
            />
          </svg>
          Not helpful
        </button>
      </div>

      {/* Inline error message */}
      {state === "error" && (
        <p className="mt-3 text-sm text-red-600">{errorMessage}</p>
      )}
    </div>
  );
}

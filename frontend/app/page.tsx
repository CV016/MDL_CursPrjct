/**
 * app/page.tsx
 * Home page — drag-and-drop document upload zone.
 *
 * Accepted formats: PDF, DOCX, PPTX.
 * On successful upload the response is saved to localStorage and the user
 * is redirected to /result where the summary and questions are displayed.
 */

"use client";

import { useCallback, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/services/api";

type UploadState = "idle" | "dragging" | "uploading" | "error";

// ---------------------------------------------------------------------------
// Module-level constants — defined outside the component so they are stable
// references and do not need to be listed in useCallback dependency arrays.
// ---------------------------------------------------------------------------

const ACCEPTED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
];

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".pptx"];

const FILE_SIZE_LIMIT_BYTES = 50 * 1024 * 1024; // 50 MB

/**
 * Validate a candidate file against accepted types and size limits.
 * Defined outside the component so it is a stable reference — this avoids
 * violating the react-hooks/exhaustive-deps rule when it is referenced inside
 * useCallback hooks.
 */
function validateFile(file: File): string | null {
  const typeOk = ACCEPTED_TYPES.includes(file.type);
  const extOk = ACCEPTED_EXTENSIONS.some((ext) => file.name.endsWith(ext));
  if (!typeOk && !extOk) {
    return "Unsupported file type. Please upload a PDF, DOCX, or PPTX file.";
  }
  if (file.size > FILE_SIZE_LIMIT_BYTES) {
    return "File is too large. Maximum allowed size is 50 MB.";
  }
  return null;
}

export default function HomePage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  /* -------------------------------------------------------------------------
   * Drag-and-drop handlers
   * ---------------------------------------------------------------------- */
  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setUploadState("dragging");
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    // Only reset the visual drag state when the cursor leaves the drop zone
    // boundary entirely — not when it moves onto a child element inside it.
    // relatedTarget is the element being entered; if it is still inside the
    // drop zone, the leave event should be ignored.
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setUploadState("idle");
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setUploadState("idle");
    const file = e.dataTransfer.files?.[0] ?? null;
    if (!file) return;
    const err = validateFile(file);
    if (err) {
      // Clear any previously selected file so the submit button is disabled
      // while an error is displayed — avoids the inconsistent state where an
      // error message is visible but the old valid file is still queued.
      setSelectedFile(null);
      setErrorMessage(err);
      setUploadState("error");
      return;
    }
    setSelectedFile(file);
    setErrorMessage("");
  }, []); // validateFile is stable (module-level function) — no dep needed

  /* -------------------------------------------------------------------------
   * File input (click-to-browse) handler
   * ---------------------------------------------------------------------- */
  const onFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (!file) return;
    const err = validateFile(file);
    if (err) {
      setSelectedFile(null);
      setErrorMessage(err);
      setUploadState("error");
      return;
    }
    setSelectedFile(file);
    setErrorMessage("");
    setUploadState("idle");
  }, []); // validateFile is stable (module-level function) — no dep needed

  /* -------------------------------------------------------------------------
   * Upload submission
   * ---------------------------------------------------------------------- */
  const handleSubmit = useCallback(async () => {
    if (!selectedFile) return;
    setUploadState("uploading");
    setErrorMessage("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const data = await apiFetch("/upload", {
        method: "POST",
        body: formData,
        // Do not set Content-Type manually — the browser must set it with the
        // multipart boundary automatically when the body is FormData.
      });

      // Persist result for the result page
      localStorage.setItem("ai_doc_result", JSON.stringify(data));
      router.push("/result");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Upload failed. Please try again.";
      setErrorMessage(message);
      setUploadState("error");
    }
  }, [selectedFile, router]);

  /* -------------------------------------------------------------------------
   * Derived UI state
   * ---------------------------------------------------------------------- */
  const isDragging = uploadState === "dragging";
  const isUploading = uploadState === "uploading";

  return (
    <div className="flex flex-col items-center gap-10">
      {/* Page header */}
      <div className="text-center max-w-xl">
        <h1 className="text-4xl font-bold text-slate-900 mb-3">
          Intelligent Document Analysis
        </h1>
        <p className="text-slate-500 text-base leading-relaxed">
          Upload a PDF, Word document, or PowerPoint presentation. AI-DOC INTERACT
          will extract the text, summarise the content, and generate comprehension
          questions — all powered by production-grade MLOps.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={[
          "w-full max-w-2xl rounded-2xl border-2 border-dashed cursor-pointer",
          "flex flex-col items-center justify-center gap-4 p-14 transition-all duration-200",
          isDragging
            ? "border-brand-500 bg-brand-50"
            : "border-slate-300 bg-white hover:border-brand-400 hover:bg-slate-50",
        ].join(" ")}
      >
        {/* Upload icon */}
        <div className="rounded-full bg-brand-50 p-4">
          <svg
            className="h-10 w-10 text-brand-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
        </div>

        {selectedFile ? (
          <div className="text-center">
            <p className="text-sm font-medium text-slate-700">{selectedFile.name}</p>
            <p className="text-xs text-slate-400 mt-1">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB — click to change
            </p>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-base font-medium text-slate-700">
              {isDragging ? "Drop your file here" : "Drag and drop your document"}
            </p>
            <p className="text-sm text-slate-400 mt-1">
              or click to browse — PDF, DOCX, PPTX up to 50 MB
            </p>
          </div>
        )}

        {/* Hidden file input */}
        <input 
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.pptx"
          className="hidden"
          onChange={onFileInputChange}
        />
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="w-full max-w-2xl rounded-lg border border-red-200 bg-red-50 px-5 py-3 text-sm text-red-700">
          {errorMessage}
        </div>
      )}

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={!selectedFile || isUploading}
        className="btn-primary w-full max-w-2xl py-3 text-base"
      >
        {isUploading ? (
          <span className="flex items-center gap-2">
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Analysing document…
          </span>
        ) : (
          "Analyse Document"
        )}
      </button>
    </div>
  );
}

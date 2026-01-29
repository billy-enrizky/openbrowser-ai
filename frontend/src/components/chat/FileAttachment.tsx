"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  FileSpreadsheet,
  FileJson,
  FileCode,
  FileImage,
  File,
  Download,
  ExternalLink,
  Eye,
  X,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { FileAttachment as FileAttachmentType, FileType } from "@/types";

interface FileAttachmentProps {
  file: FileAttachmentType;
}

// Get icon based on file type
function getFileIcon(type: FileType) {
  switch (type) {
    case "csv":
      return FileSpreadsheet;
    case "json":
      return FileJson;
    case "code":
      return FileCode;
    case "image":
      return FileImage;
    case "text":
      return FileText;
    default:
      return File;
  }
}

// Get file type from filename
export function getFileTypeFromName(filename: string): FileType {
  const ext = filename.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "csv":
      return "csv";
    case "json":
      return "json";
    case "txt":
    case "md":
    case "log":
      return "text";
    case "png":
    case "jpg":
    case "jpeg":
    case "gif":
    case "webp":
    case "svg":
      return "image";
    case "pdf":
      return "pdf";
    case "js":
    case "ts":
    case "tsx":
    case "jsx":
    case "py":
    case "html":
    case "css":
    case "scss":
    case "yaml":
    case "yml":
    case "xml":
    case "sh":
    case "bash":
      return "code";
    default:
      return "unknown";
  }
}

// Format file size
function formatFileSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Parse CSV helper function
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

// Parse CSV content into headers and rows
function parseCSVContent(content: string): { headers: string[]; rows: string[][] } {
  const lines = content.split("\n").filter(line => line.trim());
  const headers = lines[0] ? parseCSVLine(lines[0]) : [];
  const rows = lines.slice(1).map(line => parseCSVLine(line));
  return { headers, rows };
}

// CSV Preview Component (for collapsed view)
function CSVPreview({ content }: { content: string }) {
  const { headers, rows } = parseCSVContent(content);
  const maxRows = 10;
  const displayRows = rows.slice(0, maxRows);
  const hasMore = rows.length > maxRows;

  return (
    <div>
      {/* Scroll container */}
      <div 
        className="w-full overflow-x-auto"
        style={{ overflowX: 'auto' }}
      >
        {/* Table with min-width to force horizontal scroll when needed */}
        <table className="text-sm w-full" style={{ minWidth: 'max-content' }}>
          <thead>
            <tr className="border-b border-zinc-700/50">
              {headers.map((header, i) => (
                <th
                  key={i}
                  className="px-3 py-2 text-left text-xs font-semibold text-cyan-400 bg-zinc-800/50 whitespace-nowrap"
                >
                  {header || `Column ${i + 1}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
              >
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="px-3 py-2 text-zinc-300 whitespace-nowrap"
                    title={cell}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <div className="px-3 py-2 text-xs text-zinc-500 bg-zinc-800/30">
          ... and {rows.length - maxRows} more rows (scroll horizontally to see all columns)
        </div>
      )}
    </div>
  );
}

// CSV Table Component (for expanded inline table view)
interface CSVTableProps {
  content: string;
  onDownload: () => void;
}

function CSVTable({ content, onDownload }: CSVTableProps) {
  const { headers, rows } = parseCSVContent(content);

  return (
    <div className="border-t border-zinc-700/50 bg-zinc-900/50">
      {/* Download button bar */}
      <div className="flex justify-end items-center px-4 py-2 bg-zinc-800/30 border-b border-zinc-700/30">
        <button
          onClick={onDownload}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 hover:text-cyan-300 text-sm font-medium transition-colors"
        >
          <Download size={14} />
          Download CSV
        </button>
      </div>
      
      {/* Scroll container - handles both horizontal and vertical scrolling */}
      <div 
        className="max-h-[400px] overflow-auto"
        style={{
          overflowX: 'auto',
          overflowY: 'auto',
        }}
      >
        {/* Table with min-width to force horizontal scroll when needed */}
        <table className="w-full" style={{ minWidth: 'max-content' }}>
          <thead className="bg-zinc-800/80 sticky top-0 z-10">
            <tr>
              {headers.map((header, i) => (
                <th
                  key={i}
                  className="px-4 py-3 text-left text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-700/50 whitespace-nowrap"
                >
                  {header || `Column ${i + 1}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {rows.map((row, rowIndex) => (
              <tr 
                key={rowIndex} 
                className="hover:bg-zinc-800/30 transition-colors"
              >
                {headers.map((_, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="px-4 py-2.5 text-sm text-zinc-300 border-b border-zinc-800/30 whitespace-nowrap"
                  >
                    {row[cellIndex] || ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > 0 && (
        <p className="text-xs text-zinc-500 py-2 text-center bg-zinc-800/30 border-t border-zinc-700/30">
          Showing {rows.length} rows - {headers.length} columns - Scroll horizontally to see all data
        </p>
      )}
    </div>
  );
}

// JSON Preview Component
function JSONPreview({ content }: { content: string }) {
  let formatted: string;
  try {
    const parsed = JSON.parse(content);
    formatted = JSON.stringify(parsed, null, 2);
  } catch {
    formatted = content;
  }

  const lines = formatted.split("\n");
  const maxLines = 20;
  const displayContent = lines.slice(0, maxLines).join("\n");
  const hasMore = lines.length > maxLines;

  return (
    <div className="overflow-x-auto">
      <pre className="text-sm text-cyan-300 font-mono p-3">
        {displayContent}
        {hasMore && (
          <span className="text-zinc-500">
            {"\n"}... and {lines.length - maxLines} more lines
          </span>
        )}
      </pre>
    </div>
  );
}

// Code Preview Component
function CodePreview({ content, filename }: { content: string; filename: string }) {
  const lines = content.split("\n");
  const maxLines = 30;
  const displayContent = lines.slice(0, maxLines).join("\n");
  const hasMore = lines.length > maxLines;

  return (
    <div className="overflow-x-auto">
      <pre className="text-sm text-zinc-300 font-mono p-3">
        {displayContent}
        {hasMore && (
          <span className="text-zinc-500">
            {"\n"}... and {lines.length - maxLines} more lines
          </span>
        )}
      </pre>
    </div>
  );
}

// Text Preview Component
function TextPreview({ content }: { content: string }) {
  const maxChars = 2000;
  const displayContent = content.slice(0, maxChars);
  const hasMore = content.length > maxChars;

  return (
    <div className="p-3 text-sm text-zinc-300 whitespace-pre-wrap break-words">
      {displayContent}
      {hasMore && (
        <span className="text-zinc-500">
          ... and {content.length - maxChars} more characters
        </span>
      )}
    </div>
  );
}

// Image Preview Component
function ImagePreview({ content, name }: { content: string; name: string }) {
  const src = content.startsWith("data:") ? content : `data:image/png;base64,${content}`;
  
  return (
    <div className="p-3 flex justify-center">
      <img
        src={src}
        alt={name}
        className="max-w-full max-h-[400px] rounded-lg object-contain"
      />
    </div>
  );
}

export function FileAttachment({ file }: FileAttachmentProps) {
  // All files are collapsed by default - user can expand them
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const Icon = getFileIcon(file.type);

  const handleDownload = useCallback(() => {
    // Prefer content-based download over URL (file:// URLs don't work in browser)
    if (file.content) {
      // Create blob and download
      let blob: Blob;
      let mimeType = file.mimeType || "text/plain";
      
      // Set appropriate MIME type based on file type
      if (file.type === "csv") {
        mimeType = "text/csv";
      } else if (file.type === "json") {
        mimeType = "application/json";
      } else if (file.type === "image") {
        // Handle base64 image
        const byteCharacters = atob(file.content.replace(/^data:.*?;base64,/, ""));
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        blob = new Blob([byteArray], { type: mimeType || "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = file.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return;
      }
      
      blob = new Blob([file.content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return;
    }
    
    // Fallback to URL if no content and URL is valid (not file://)
    if (file.url && !file.url.startsWith("file://")) {
      window.open(file.url, "_blank");
    }
  }, [file]);

  const handleOpenInNewTab = useCallback(() => {
    // Prefer content-based view over URL (file:// URLs don't work in browser)
    if (file.content) {
      // Create blob URL and open
      let blob: Blob;
      let mimeType = file.mimeType || "text/plain";
      
      if (file.type === "csv") {
        mimeType = "text/csv";
      } else if (file.type === "json") {
        mimeType = "application/json";
      } else if (file.type === "image") {
        const base64Content = file.content.replace(/^data:.*?;base64,/, "");
        const byteCharacters = atob(base64Content);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        blob = new Blob([byteArray], { type: mimeType });
        const url = URL.createObjectURL(blob);
        window.open(url, "_blank");
        return;
      }
      
      blob = new Blob([file.content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
      return;
    }
    
    // Fallback to URL if no content and URL is valid (not file://)
    if (file.url && !file.url.startsWith("file://")) {
      window.open(file.url, "_blank");
    }
  }, [file]);

  const handleCopy = useCallback(async () => {
    if (file.content) {
      await navigator.clipboard.writeText(file.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [file.content]);

  const canPreview = file.content && ["csv", "json", "text", "code", "image"].includes(file.type);

  // For CSV files, render a collapsible component with table inside
  if (file.type === "csv" && file.content) {
    const { rows } = parseCSVContent(file.content);
    return (
      <div className="rounded-xl border border-zinc-700/50 bg-zinc-800/30 overflow-hidden">
        {/* Collapsible Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-zinc-800/50">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-3 min-w-0 hover:opacity-80 transition-opacity"
          >
            <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 bg-green-500/20 text-green-400">
              <FileSpreadsheet className="w-5 h-5" />
            </div>
            <div className="min-w-0 text-left">
              <div className="text-sm font-medium text-zinc-200 truncate">
                {file.name}
              </div>
              <div className="text-xs text-zinc-500">
                CSV - {rows.length} rows
                {file.size && ` - ${formatFileSize(file.size)}`}
              </div>
            </div>
          </button>

          <div className="flex items-center gap-2">
            {/* Download button - always visible */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDownload();
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 hover:text-cyan-300 text-sm font-medium transition-colors"
              title="Download CSV"
            >
              <Download size={14} />
              <span className="hidden sm:inline">Download</span>
            </button>
            
            {/* Expand/Collapse button */}
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors"
              title={isExpanded ? "Collapse" : "Expand to view"}
            >
              <span className="text-xs hidden sm:inline">
                {isExpanded ? "Collapse" : "Expand"}
              </span>
              {isExpanded ? (
                <ChevronUp className="w-5 h-5" />
              ) : (
                <ChevronDown className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* Expandable CSV Table */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              style={{ overflow: 'hidden' }}
            >
              <CSVTable 
                content={file.content} 
                onDownload={handleDownload} 
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-zinc-700/50 bg-zinc-800/30 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-zinc-800/50">
        <div className="flex items-center gap-3 min-w-0">
          <div className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center shrink-0",
            file.type === "csv" && "bg-green-500/20 text-green-400",
            file.type === "json" && "bg-yellow-500/20 text-yellow-400",
            file.type === "code" && "bg-blue-500/20 text-blue-400",
            file.type === "image" && "bg-purple-500/20 text-purple-400",
            file.type === "text" && "bg-zinc-500/20 text-zinc-400",
            file.type === "unknown" && "bg-zinc-500/20 text-zinc-400"
          )}>
            <Icon className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium text-zinc-200 truncate">
              {file.name}
            </div>
            <div className="text-xs text-zinc-500">
              {file.type.toUpperCase()}
              {file.size && ` - ${formatFileSize(file.size)}`}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {canPreview && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors"
              title={isExpanded ? "Collapse preview" : "Expand preview"}
            >
              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          )}
          {file.content && file.type !== "image" && (
            <button
              onClick={handleCopy}
              className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors"
              title="Copy content"
            >
              {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
            </button>
          )}
          <button
            onClick={handleOpenInNewTab}
            className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors"
            title="Open in new tab"
          >
            <ExternalLink className="w-4 h-4" />
          </button>
          <button
            onClick={handleDownload}
            className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors"
            title="Download"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Preview */}
      <AnimatePresence>
        {isExpanded && canPreview && file.content && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-zinc-700/50 bg-zinc-900/50 max-h-[400px] overflow-auto"
          >
            {file.type === "csv" && <CSVPreview content={file.content} />}
            {file.type === "json" && <JSONPreview content={file.content} />}
            {file.type === "code" && <CodePreview content={file.content} filename={file.name} />}
            {file.type === "text" && <TextPreview content={file.content} />}
            {file.type === "image" && <ImagePreview content={file.content} name={file.name} />}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// File Preview Modal Component
interface FilePreviewModalProps {
  file: FileAttachmentType | null;
  onClose: () => void;
}

export function FilePreviewModal({ file, onClose }: FilePreviewModalProps) {
  if (!file) return null;

  const Icon = getFileIcon(file.type);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="w-full max-w-4xl max-h-[90vh] bg-zinc-900 rounded-2xl border border-zinc-700/50 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-700/50">
            <div className="flex items-center gap-3">
              <Icon className="w-5 h-5 text-zinc-400" />
              <span className="font-medium text-zinc-200">{file.name}</span>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Modal Content */}
          <div className="overflow-auto max-h-[calc(90vh-80px)]">
            {file.content && (
              <>
                {file.type === "csv" && <CSVPreview content={file.content} />}
                {file.type === "json" && <JSONPreview content={file.content} />}
                {file.type === "code" && <CodePreview content={file.content} filename={file.name} />}
                {file.type === "text" && <TextPreview content={file.content} />}
                {file.type === "image" && <ImagePreview content={file.content} name={file.name} />}
              </>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


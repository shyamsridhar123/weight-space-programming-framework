import http from "node:http";
import { createReadStream } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DECK_ROOT = path.dirname(fileURLToPath(import.meta.url));
const MIME = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".map": "application/json; charset=utf-8",
  ".ts": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8", ".vtt": "text/vtt; charset=utf-8",
  ".mp4": "video/mp4", ".webm": "video/webm", ".mp3": "audio/mpeg",
  ".wav": "audio/wav", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".png": "image/png", ".webp": "image/webp", ".svg": "image/svg+xml",
  ".ico": "image/x-icon", ".woff": "font/woff", ".woff2": "font/woff2",
};

function sendText(response, status, message, head = false, extra = {}) {
  const body = Buffer.from(message + "\n");
  response.writeHead(status, { "Content-Type": "text/plain; charset=utf-8", "Content-Length": body.length, ...extra });
  response.end(head ? undefined : body);
}

function within(root, target) {
  const relative = path.relative(root, target);
  return relative === "" || (!relative.startsWith(".." + path.sep) && relative !== ".." && !path.isAbsolute(relative));
}

export function byteRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header || "");
  if (!match || (!match[1] && !match[2]) || size <= 0) return null;
  let start;
  let end;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return null;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start >= size || end < start) return null;
  return { start, end };
}

export async function createDeckServer(root = DECK_ROOT) {
  const realRoot = await realpath(root);
  return http.createServer(async (request, response) => {
    response.setHeader("Access-Control-Allow-Origin", "*");
    response.setHeader("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
    response.setHeader("Access-Control-Allow-Headers", "Range");
    response.setHeader("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges");
    if (request.method === "OPTIONS") {
      response.writeHead(204);
      response.end();
      return;
    }
    const head = request.method === "HEAD";
    if (request.method !== "GET" && !head) {
      sendText(response, 405, "Use GET or HEAD.", head, { Allow: "GET, HEAD, OPTIONS" });
      return;
    }
    try {
      const rawPath = (request.url || "/").split("?")[0];
      const pathname = decodeURIComponent(rawPath);
      if (!pathname.startsWith("/") || pathname.includes("\\") || pathname.includes("\0") || pathname.includes(":") ||
        pathname.split("/").some((part) => part.startsWith(".") || part === "node_modules")) {
        sendText(response, 403, "This path is not available.", head);
        return;
      }
      let filename = path.resolve(realRoot, "." + pathname);
      if (!within(realRoot, filename)) { sendText(response, 403, "This path is not available.", head); return; }
      let info = await stat(filename);
      if (info.isDirectory()) {
        if (!pathname.endsWith("/")) {
          response.writeHead(308, { Location: encodeURI(pathname + "/") });
          response.end();
          return;
        }
        filename = path.join(filename, "index.html");
        info = await stat(filename);
      }
      filename = await realpath(filename);
      if (!within(realRoot, filename) || !info.isFile()) { sendText(response, 403, "This path is not available.", head); return; }

      const headers = {
        "Content-Type": MIME[path.extname(filename).toLowerCase()] || "application/octet-stream",
        "Accept-Ranges": "bytes", "Cache-Control": "no-cache",
        "Last-Modified": info.mtime.toUTCString(), "X-Content-Type-Options": "nosniff",
      };
      const range = request.headers.range ? byteRange(request.headers.range, info.size) : undefined;
      if (range === null) {
        sendText(response, 416, "Requested byte range is unavailable.", head, { "Content-Range": `bytes */${info.size}` });
        return;
      }
      const start = range?.start ?? 0;
      const end = range?.end ?? info.size - 1;
      headers["Content-Length"] = Math.max(0, end - start + 1);
      if (range) headers["Content-Range"] = `bytes ${start}-${end}/${info.size}`;
      response.writeHead(range ? 206 : 200, headers);
      if (head || info.size === 0) { response.end(); return; }
      const stream = createReadStream(filename, { start, end });
      stream.on("error", () => response.destroy());
      response.on("close", () => stream.destroy());
      stream.pipe(response);
    } catch (error) {
      if (response.headersSent) { response.destroy(); return; }
      const missing = error?.code === "ENOENT" || error?.code === "ENOTDIR";
      sendText(response, error instanceof URIError ? 400 : missing ? 404 : 500,
        error instanceof URIError ? "Invalid URL." : missing ? "File not found." : "Could not serve this file.", head);
    }
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const value = process.argv[2] || process.env.PORT || "4180";
  if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > 65535) {
    console.error("Choose a port between 1 and 65535: node serve.mjs 4180");
    process.exitCode = 1;
  } else {
    const port = Number(value);
    const server = await createDeckServer();
    server.on("error", (error) => {
      console.error(error.code === "EADDRINUSE"
        ? `Port ${port} is already in use. Try: node serve.mjs ${port + 1}`
        : error.message);
      process.exitCode = 1;
    });
    server.listen(port, "127.0.0.1", () => {
      console.log(`Crown & Charter presentation: http://127.0.0.1:${port}/`);
      console.log("Open that address in your browser. Keep this window open; Ctrl+C stops the server.");
    });
    process.once("SIGINT", () => server.close());
    process.once("SIGTERM", () => server.close());
  }
}

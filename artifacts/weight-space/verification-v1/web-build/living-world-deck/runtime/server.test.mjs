import assert from "node:assert/strict";
import { request } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { once } from "node:events";
import path from "node:path";
import { after, before, test } from "node:test";
import { createDeckServer, DECK_ROOT } from "../serve.mjs";

let server;
let port;
before(async () => {
  server = await createDeckServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  port = server.address().port;
});
after(async () => {
  if (!server?.listening) return;
  const closed = once(server, "close");
  server.close();
  await closed;
});

function get(url, { method = "GET", headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const req = request({ host: "127.0.0.1", port, path: url, method, headers }, (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => resolve({ status: response.statusCode, headers: response.headers, body: Buffer.concat(chunks) }));
      response.on("error", reject);
    });
    req.on("error", reject);
    req.end();
  });
}

test("serves the portable wrapper and same-origin assets", async () => {
  const response = await get("/");
  assert.equal(response.status, 200);
  assert.match(response.headers["content-type"], /^text\/html/);
  assert.match(response.body.toString(), /<hyperframes-slideshow/);
  const script = await get("/runtime/composition-runtime.js");
  assert.equal(script.status, 200);
  assert.match(script.headers["content-type"], /^text\/javascript/);
  const redirect = await get("/composition");
  assert.equal(redirect.status, 308);
  assert.equal(redirect.headers.location, "/composition/");
});

test("video seeking returns exact bounded, open, and suffix byte ranges", async () => {
  const filename = path.join(DECK_ROOT, "assets", "gameplay-timelapse.mp4");
  const bytes = await readFile(filename);
  for (const [range, start, end] of [
    ["bytes=0-63", 0, 63],
    [`bytes=${bytes.length - 32}-`, bytes.length - 32, bytes.length - 1],
    ["bytes=-48", bytes.length - 48, bytes.length - 1],
    [`bytes=${bytes.length - 16}-${bytes.length + 50}`, bytes.length - 16, bytes.length - 1],
  ]) {
    const response = await get("/assets/gameplay-timelapse.mp4", { headers: { Range: range } });
    assert.equal(response.status, 206, range);
    assert.equal(response.headers["content-type"], "video/mp4");
    assert.equal(response.headers["content-range"], `bytes ${start}-${end}/${bytes.length}`);
    assert.equal(Number(response.headers["content-length"]), end - start + 1);
    assert.deepEqual(response.body, bytes.subarray(start, end + 1));
  }
});

test("HEAD, captions, readable sources, and CORS retain the right headers", async () => {
  const info = await stat(path.join(DECK_ROOT, "assets", "gameplay-timelapse.mp4"));
  const head = await get("/assets/gameplay-timelapse.mp4", { method: "HEAD" });
  assert.equal(head.status, 200);
  assert.equal(Number(head.headers["content-length"]), info.size);
  assert.equal(head.body.length, 0);
  assert.equal(head.headers["accept-ranges"], "bytes");
  assert.equal(head.headers["access-control-allow-origin"], "*");
  const captions = await get("/assets/gameplay-captions.vtt");
  assert.equal(captions.status, 200);
  assert.match(captions.headers["content-type"], /^text\/vtt/);
  assert.match(captions.body.toString(), /^WEBVTT/);
  const source = await get("/sources/policy.ts");
  assert.equal(source.status, 200);
  assert.match(source.headers["content-type"], /^text\/plain/);
  const options = await get("/assets/gameplay-timelapse.mp4", { method: "OPTIONS" });
  assert.equal(options.status, 204);
  assert.match(options.headers["access-control-allow-headers"], /Range/);
});

test("invalid byte ranges fail explicitly, without returning the full video", async () => {
  for (const range of ["bytes=999999999999-", "bytes=50-10", "bytes=-0", "bytes=0-1,4-5", "nonsense"]) {
    const response = await get("/assets/gameplay-timelapse.mp4", { headers: { Range: range } });
    assert.equal(response.status, 416, range);
    assert.match(response.headers["content-range"], /^bytes \*\/\d+$/);
    assert.ok(response.body.length < 1000);
  }
});

test("missing, malformed, hidden, and traversal paths do not expose workspace files", async () => {
  for (const url of ["/%2e%2e/package.json", "/.bootstrap/index.html", "/node_modules/pkg/index.js", "/a%5cb", "/index.html:stream"]) {
    assert.equal((await get(url)).status, 403, url);
  }
  assert.equal((await get("/missing.html")).status, 404);
  assert.equal((await get("/%ZZ")).status, 400);
  const post = await get("/", { method: "POST" });
  assert.equal(post.status, 405);
});

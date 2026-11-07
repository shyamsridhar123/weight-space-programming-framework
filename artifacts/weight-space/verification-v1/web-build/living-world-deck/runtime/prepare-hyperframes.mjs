import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { syncManifest } from "./sync-manifest.mjs";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const stage = path.join(root, ".hyperframes-project");
await syncManifest();
await mkdir(stage, { recursive: true });

// The CLI expects a raw composition at index.html; the portable entry page is
// already an official slideshow wrapper. Stage the raw page without changing it.
const composition = await readFile(path.join(root, "composition", "index.html"), "utf8");
await writeFile(path.join(stage, "index.html"), composition.replace(
  /(["'])\.\.\/(assets|lab|runtime|vendor|sources)\//g, "$1$2/",
));
await cp(path.join(root, "composition", "deck.css"), path.join(stage, "deck.css"));
for (const name of ["assets", "lab", "runtime", "vendor", "sources"]) {
  await cp(path.join(root, name), path.join(stage, name), { recursive: true, force: true });
}
await cp(path.join(root, "hyperframes.json"), path.join(stage, "hyperframes.json"));
await writeFile(path.join(stage, "package.json"), JSON.stringify({
  name: "crown-and-charter-presentation", private: true, type: "module",
}, null, 2) + "\n");
console.log("HyperFrames composition staged in .hyperframes-project/");

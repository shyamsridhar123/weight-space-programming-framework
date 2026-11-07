import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const manifestPattern = /<script\s+type="application\/hyperframes-slideshow\+json">[\s\S]*?<\/script>/;

export async function syncManifest() {
  const composition = await readFile(path.join(root, "composition", "index.html"), "utf8");
  const filename = path.join(root, "index.html");
  const wrapper = await readFile(filename, "utf8");
  const manifest = composition.match(manifestPattern)?.[0];
  if (!manifest || !manifestPattern.test(wrapper)) throw new Error("Both entry pages need a slideshow manifest.");
  JSON.parse(manifest.slice(manifest.indexOf(">") + 1, manifest.lastIndexOf("<")));
  const updated = wrapper.replace(manifestPattern, () => manifest);
  if (updated !== wrapper) await writeFile(filename, updated);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await syncManifest();
  console.log("Wrapper speaker notes synchronized from composition/index.html.");
}

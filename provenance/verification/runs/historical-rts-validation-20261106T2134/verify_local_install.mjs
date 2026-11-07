import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const staging = realpathSync(process.cwd());
const localModules = realpathSync(path.join(staging, 'node_modules'));
const stagedRequire = createRequire(path.join(staging, 'package.json'));
const packages = ['tsx', 'esbuild', '@esbuild/win32-x64'].map(name => {
  const resolved = realpathSync(stagedRequire.resolve(`${name}/package.json`));
  const relative = path.relative(localModules, resolved);
  assert.ok(relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), `${name} must resolve inside staging/node_modules`);
  const bytes = readFileSync(resolved);
  const metadata = JSON.parse(bytes.toString('utf8'));
  return { name, version: metadata.version, resolved, relative_to_staged_node_modules: relative, sha256: createHash('sha256').update(bytes).digest('hex') };
});

const launcher = path.join(localModules, '.bin', 'tsx.cmd');
assert.ok(existsSync(launcher));
const launcherText = readFileSync(launcher, 'utf8');
assert.ok(launcherText.includes('..\\tsx\\dist\\cli.mjs'), 'npm launcher must use staged tsx CLI');
const cli = path.join(localModules, 'tsx', 'dist', 'cli.mjs');
const version = spawnSync(process.execPath, [cli, '--version'], { cwd: staging, encoding: 'utf8' });
assert.equal(version.status, 0, version.stderr);
const esbuild = stagedRequire('esbuild');
const transform = esbuild.transformSync('const value: number = 1;', { loader: 'ts' });
assert.ok(transform.code.includes('const value = 1;'));
console.log(JSON.stringify({
  staging,
  node: { executable: process.execPath, version: process.version },
  node_path: process.env.NODE_PATH ?? null,
  packages,
  tsx_launcher: { path: launcher, sha256: createHash('sha256').update(launcherText).digest('hex') },
  tsx_version_command: { argv: [process.execPath, cli, '--version'], exit_code: version.status, stdout: version.stdout, stderr: version.stderr },
  local_esbuild_transform: { code: transform.code, warnings: transform.warnings },
  all_dependency_paths_inside_staging: true
}, null, 2));

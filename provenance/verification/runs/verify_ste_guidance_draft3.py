"""Verify pinned guidance bytes and preserve the one extra-newline reading copy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDANCE = ROOT / '.runs/ste-reference'
source = json.loads((GUIDANCE / 'source.json').read_text(encoding='utf-8-sig'))
expected = '0607fda2c4ad5e036ce54eaedcc10d293679c0e7'
reading = GUIDANCE / 'skills/ste-line-studio/references/authoring.md'
retrieved = GUIDANCE / 'authoring-exact-github.md.tmp'
preserved = GUIDANCE / 'authoring-reading-copy-before-byte-verification.md.txt'
output = GUIDANCE / 'source-byte-verification-draft3.json'

def blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()

def bind(path, raw):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob_sha': blob(raw)}

if preserved.exists() or output.exists():
    raise FileExistsError('Refusing to overwrite a completed guidance-byte correction')
before = reading.read_bytes()
exact = retrieved.read_bytes()
if blob(exact) != expected or before != exact + b'\n':
    raise ValueError('Fetched blob or narrowly understood trailing-newline difference is unexpected')
records = []
for item in source['files']:
    path = GUIDANCE / item['path']
    raw = exact if path == reading else path.read_bytes()
    if blob(raw) != item['blob_sha']:
        raise ValueError(f'Another retrieved guidance file differs: {item["path"]}')
    records.append({'repository_path': item['path'], 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob_sha': blob(raw)})
with preserved.open('xb') as stream:
    stream.write(before)
reading.write_bytes(exact)
record = {
    'schema': 'wspf-guidance-byte-verification/1',
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'passed',
    'repository': source['repository'], 'commit': source['commit'],
    'acquisition': 'Authenticated GitHub fetch_blob for the pinned authoring.md blob; returned UTF-8 content was saved and independently checked against the Git blob identifier.',
    'correction': {
        'difference': 'The local reading copy had exactly one additional trailing LF byte; no other content differed.',
        'before': bind(reading, before),
        'preserved_local_reading_copy': bind(preserved, before),
        'after': bind(reading, exact),
        'substantive_guidance_changed': False,
    },
    'files': records,
    'limitations': ['Content identity does not attest execution of any STE runtime or establish documentation quality.'],
}
with output.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(record, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
print(json.dumps({'status': 'passed', 'record': output.relative_to(ROOT).as_posix(),
                  'correction': record['correction']['difference']}))

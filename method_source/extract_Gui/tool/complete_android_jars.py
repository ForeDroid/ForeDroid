#!/usr/bin/env python3
"""Fill missing API 18..36 android.jar files from Google's platform archives."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

BASE = 'https://dl.google.com/android/repository/'
INDEX = BASE + 'repository2-1.xml'

def validate(data):
    with zipfile.ZipFile(io.BytesIO(data)) as jar:
        if jar.testzip() is not None or 'android/app/Activity.class' not in jar.namelist():
            raise ValueError('Invalid Android platform JAR')

def main():
    dest = Path(__file__).resolve().parents[1] / 'android-18/android_jars'
    dest.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(urllib.request.urlopen(INDEX, timeout=60).read())
    manifest = {}
    for api in range(18, 37):
        path = dest / f'android-{api}.jar'
        if path.exists():
            data = path.read_bytes()
            validate(data)
            manifest[str(api)] = {'status': 'existing', 'sha256': hashlib.sha256(data).hexdigest()}
            print(f'API {api}: existing, verified ZIP/classes', flush=True)
            continue
        candidates = []
        for p in root.findall('remotePackage'):
            if p.get('path') != f'platforms;android-{api}':
                continue
            for a in p.findall('archives/archive'):
                url = a.findtext('complete/url')
                if not url or a.findtext('host-os') not in (None, 'linux'):
                    continue
                # Prefer base platform over SDK extension variants, then latest revision.
                rev = tuple(int(p.findtext('revision/'+k) or 0) for k in ('major','minor','micro'))
                candidates.append(('-ext' not in url, rev, url, a.findtext('complete/checksum'), int(a.findtext('complete/size'))))
        if not candidates:
            raise RuntimeError(f'Official platform archive not found: API {api}')
        _, _, name, checksum, size = max(candidates)
        print(f'API {api}: downloading {name}', flush=True)
        with tempfile.TemporaryFile() as archive:
            digest = hashlib.sha1()
            count = 0
            with urllib.request.urlopen(BASE+name, timeout=120) as response:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    archive.write(chunk); digest.update(chunk); count += len(chunk)
            if digest.hexdigest() != checksum or count != size:
                raise ValueError(f'Archive checksum/size mismatch: {name}')
            archive.seek(0)
            with zipfile.ZipFile(archive) as z:
                jars = [n for n in z.namelist() if n.endswith('/android.jar')]
                if len(jars) != 1:
                    raise ValueError(f'Ambiguous android.jar in {name}')
                prefix = jars[0][:-len('android.jar')]
                props = z.read(prefix+'source.properties').decode()
                parsed = dict(line.split('=',1) for line in props.splitlines() if '=' in line)
                if parsed.get('AndroidVersion.ApiLevel','').strip() != str(api):
                    raise ValueError('Archive API mismatch')
                data = z.read(jars[0])
        validate(data)
        pending = path.with_suffix('.jar.part')
        pending.write_bytes(data)
        pending.replace(path)
        manifest[str(api)] = {'status': 'downloaded', 'url': BASE+name,
            'archive_sha1': checksum, 'sha256': hashlib.sha256(data).hexdigest(),
            'source_properties': props}
        print(f'API {api}: installed ({len(data)} bytes)', flush=True)
    report = dest / 'platform-verification.json'
    report.write_text(json.dumps({'repository':INDEX,'platforms':manifest},indent=2)+'\n')
    print('COMPLETE: all 19 API versions (18..36) verified',flush=True)

if __name__ == '__main__': main()

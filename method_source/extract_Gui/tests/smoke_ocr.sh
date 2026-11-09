#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Exercise the same Java adapter called directly by extract_ui.sh.
ocr_export() {
 java -Djava.io.tmpdir="$work/scratch" --add-opens java.base/java.util=ALL-UNNAMED \
  --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
  --add-opens java.base/java.text=ALL-UNNAMED \
  --add-opens java.desktop/java.awt.font=ALL-UNNAMED \
  -cp ocr/build:jars/Backstage-UIAnalysis.jar OcrUIAnalysis \
  "$1" "$2" "$PWD/ocr/ocr_assets.py" python3
}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/scratch"
export TMPDIR="$work/scratch"
java --add-opens java.base/java.util=ALL-UNNAMED \
 --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
 --add-opens java.base/java.text=ALL-UNNAMED \
 --add-opens java.desktop/java.awt.font=ALL-UNNAMED \
 -cp tests/build:jars/Backstage-UIAnalysis.jar CreateOcrFixture "$work/output/fixture"
python3 - "$work" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
app=Path(sys.argv[1])/'output/fixture'
(app/'res/drawable').mkdir(parents=True)
(app/'res/values').mkdir(parents=True)
(app/'res/values/strings.xml').write_text('<resources/>')
(app/'res/values/styles.xml').write_text('<resources><style name="AppTheme"><item name="android:colorAccent">#ffffff</item></style></resources>')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',48)
for name,text in [('login','Log In'),('title','Welcome Back')]:
 im=Image.new('RGB',(600,130),'white')
 ImageDraw.Draw(im).text((30,30),text,font=font,fill='black')
 im.save(app/f'res/drawable/{name}.png')
PY
before=$(sha256sum "$work/output/fixture/appSerialized.txt")
ocr_export "$work/output" "$work/outputBenignUi_train.txt"
[[ "$before" == "$(sha256sum "$work/output/fixture/appSerialized.txt")" ]]
python3 - "$work" <<'PY'
import csv,sys
from pathlib import Path
work=Path(sys.argv[1])
with (work/'outputBenignUi_train.txt').open() as f: rows=list(csv.DictReader(f,delimiter=';'))
button=next(r for r in rows if r['uid']=='101')
assert 'log in' in button['label'].lower(),button
assert 'welcome back' in button['context'].lower(),button
assert button['hascallback']=='CALL',button
with (work/'api.csv').open('w',newline='') as f:
 w=csv.writer(f,delimiter=';');w.writerow(['apk','uid','declaringClass','callback','api'])
 w.writerow(['fixture.apk','101','example.MainActivity','<example.MainActivity: void onClick(android.view.View)>','<android.telephony.SmsManager: void sendTextMessage()>'])
PY
python3 merge_ui_api.py --ui-file "$work/outputBenignUi_train.txt" --api-file "$work/api.csv" --output-dir "$work/merge"
python3 - "$work" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1])/'merge/merge_ui_api_with_relation.txt'
with p.open() as f: rows=list(csv.DictReader(f,delimiter=';'))
assert any('Log In' in r['label'] and r['callback'] and r['relation_strength']=='STRONG' for r in rows),rows
PY
csv_before=$(sha256sum "$work/outputBenignUi_train.txt")
if TESSERACT=foredroid-nonexistent-tesseract ocr_export "$work/output" "$work/outputBenignUi_train.txt"; then
 echo 'Missing Tesseract did not fail' >&2; exit 1
fi
[[ "$csv_before" == "$(sha256sum "$work/outputBenignUi_train.txt")" ]]
[[ -z "$(find "$work/scratch" -mindepth 1 -print -quit)" ]]
[[ ! -e "$work/ocr-reports" ]]
echo 'OCR smoke test passed (train naming/original schema; no persistent OCR output)' 

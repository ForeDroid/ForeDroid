import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image

spec = importlib.util.spec_from_file_location('ocr_assets', Path(__file__).resolve().parents[1] / 'ocr/ocr_assets.py')
ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocr)

class ResourceTests(unittest.TestCase):
    def test_alias_selector_density_and_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            for folder in ['drawable', 'drawable-hdpi', 'values']:
                (app / 'res' / folder).mkdir(parents=True)
            a = app / 'res/drawable/login.png'; a.touch()
            b = app / 'res/drawable-hdpi/login.9.png'; b.touch()
            (app / 'res/drawable/selector.xml').write_text('<selector xmlns:android="http://schemas.android.com/apk/res/android"><item android:drawable="@drawable/login"/><item android:drawable="@drawable/selector"/></selector>')
            (app / 'res/values/aliases.xml').write_text('<resources><item type="drawable" name="alias">@drawable/selector</item></resources>')
            r = ocr.Resources(app)
            self.assertEqual(set(r.resolve('@drawable/alias')), {a,b})
            self.assertEqual(set(r.resolve('login.png')), {a,b})
            self.assertEqual(r.resolve('@android:drawable/login'), [])
            self.assertEqual(r.resolve('../../other.png'), [])

    def test_widget_binding_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp); (app / 'res/drawable').mkdir(parents=True)
            for name in ['login', 'unrelated']:
                (app / f'res/drawable/{name}.png').touch()
            requests = app / 'requests.tsv'
            with requests.open('w', newline='') as f:
                w = csv.writer(f, delimiter='\t'); w.writerow(['uid','drawables'])
                w.writerow(['1', json.dumps(['login.png','@drawable/login'])])
                w.writerow(['2', '[]'])
            class Fake:
                cache = {}
                paths = []
                def read(self, path):
                    self.paths.append(path)
                    return {'text': 'Log In', 'image': str(path), 'status': 'ok'}
            fake = Fake()
            ocr.enrich(app, requests, app/'out.tsv', fake)
            with (app/'out.tsv').open() as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(rows, [{'uid':'1','ocr_text':'Log In'}, {'uid':'2','ocr_text':''}])
            self.assertEqual(fake.paths, [app/'res/drawable/login.png'])
            self.assertFalse((app/'report.jsonl').exists())

    def test_confidence_does_not_accept_punctuation_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'icon.png'
            Image.new('RGB', (30, 30), 'white').save(image)
            tsv = 'conf\ttext\n79\t~\n90\tLog\n88\tIn\n20\tnoise\n'
            responses = [SimpleNamespace(stdout='List of available languages:\neng\n'),
                         SimpleNamespace(stdout=tsv)]
            with patch.object(ocr.shutil, 'which', return_value='/usr/bin/tesseract'), \
                 patch.object(ocr.subprocess, 'run', side_effect=responses):
                result = ocr.Recognizer().read(image)
            self.assertEqual(result['text'], 'Log In')

    def test_missing_engine_fails(self):
        with self.assertRaises(RuntimeError):
            ocr.Recognizer(executable='foredroid-nonexistent-tesseract')

if __name__ == '__main__': unittest.main()

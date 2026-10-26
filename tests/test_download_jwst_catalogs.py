"""Offline downloader checks: synthetic files only, never science downloads."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from astropy.table import Table

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_jwst_cluster_images.py'
spec = importlib.util.spec_from_file_location('jwst_download', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_inventory_and_existing_image_plan():
    manifest, sources = module.catalog_sources()
    assert len(sources) == 16
    assert len({s['id'] for s in sources}) == len(sources)
    assert len(list(module.products(['AS1063','M0416'], module.FILTERS, 20))) == 12
    for source in sources:
        assert source['notes'] and source['source_page'].startswith('https://')
        assert len({p['filename'] for p in source['files']}) == len(source['files'])
        assert all(p['url'].startswith('https://') for p in source['files'])
    with pytest.raises(ValueError, match='Unknown'):
        module.catalog_sources(['unknown'])


def test_dry_run_does_not_create_files(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), '--catalogs-only', '--dry-run',
                             '--output', str(tmp_path/'images')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '16 sources' in result.stdout
    assert '_drc_sci' not in result.stdout
    assert not list(tmp_path.iterdir())


def test_validation(tmp_path):
    path = tmp_path/'table.fits.part'
    Table({'ra':[64.0], 'dec':[-24.0], 'z':[2.0]}).write(path, format='fits')
    assert module.validate_catalog(path, dict(filename='table.fits',format='fits'),tmp_path)['tables'][0]['rows']==1
    bad = tmp_path/'bad.part';bad.write_text('<html>error</html>')
    with pytest.raises(ValueError, match='HTML'):
        module.validate_catalog(bad,dict(filename='x.txt',format='text'),tmp_path)
    csv = tmp_path/'x.csv.gz.part'
    with gzip.open(csv,'wb') as f:f.write(b'ra,dec,z\n64,-24,2\n')
    assert module.validate_catalog(csv,dict(filename='x.csv.gz',format='csv'),tmp_path)['tables'][0]['rows']==1
    csv.write_bytes(csv.read_bytes()[:-5])
    with pytest.raises((EOFError,OSError)):
        module.validate_catalog(csv,dict(filename='x.csv.gz',format='csv'),tmp_path)


def test_cds_schema_uses_original_name(tmp_path):
    (tmp_path/'ReadMe').write_text('''Byte-by-byte Description of file: test.dat
--------------------------------------------------------------------------------
   Bytes Format Units Label Explanations
--------------------------------------------------------------------------------
   1-  3  I3    ---   ID    Identifier
   5-  8  F4.1  ---   z     Redshift
--------------------------------------------------------------------------------
''')
    path=tmp_path/'test.dat.gz.part'
    with gzip.open(path,'wt') as f:f.write('  1  2.0\n')
    result=module.validate_catalog(path,dict(filename='test.dat.gz',format='cds'),tmp_path)
    assert result['tables']==[dict(rows=1,columns=['ID','z'])]


def test_refresh_preserves_previous_and_failed_response(tmp_path,monkeypatch):
    product=dict(filename='x.csv',format='csv',url='https://example.org/x.csv')
    dest=tmp_path/'x.csv';dest.write_text('ra,z\n64,1\n')
    first=module.download_catalog(product,tmp_path)
    def fetch(args,**kw):Path(args[args.index('--output')+1]).write_text('ra,z\n64,2\n')
    monkeypatch.setattr(module.subprocess,'run',fetch)
    second=module.download_catalog(product,tmp_path,refresh=True)
    assert first['sha256']!=second['sha256']
    assert (tmp_path/'.versions'/first['sha256']/'x.csv').read_text()=='ra,z\n64,1\n'
    def invalid(args,**kw):Path(args[args.index('--output')+1]).write_text('<html>Error</html>')
    monkeypatch.setattr(module.subprocess,'run',invalid)
    with pytest.raises(ValueError):module.download_catalog(product,tmp_path,refresh=True)
    assert dest.read_text()=='ra,z\n64,2\n'


def test_failures_are_recorded_and_remaining_files_attempted(tmp_path,monkeypatch):
    _, sources=module.catalog_sources(['buffalo'])
    attempted=[]
    def fail(product,directory,**kw):
        attempted.append(product['filename']);raise ValueError('bad table')
    monkeypatch.setattr(module,'download_catalog',fail)
    manifest,_=module.catalog_sources()
    assert not module.download_catalogs(tmp_path,sources,manifest)
    report=json.loads((tmp_path/'M0416/download_manifest.json').read_text())
    assert len(report['failures'])==len(attempted)==2
    assert not report['complete']

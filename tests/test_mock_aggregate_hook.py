import json
from types import SimpleNamespace as NS
from lenscluster.mock_validation import runner, aggregate_plots

def test_refresh_after_each_saved_recovery_and_resume(tmp_path, monkeypatch):
    config=NS(runtime=NS(realizations=2,seed=10,resume=True),mock=NS(mode='test'))
    config.validate=lambda:config
    config.to_json_dict=lambda:{}
    monkeypatch.setattr(runner,'campaign_root',lambda c:tmp_path)
    def generate(root,*args,**kwargs):
        root.mkdir(parents=True,exist_ok=True)
        paths=runner.MockPaths(root)
        paths.truth.write_text(json.dumps({'z_lens':.4,'halo':{'v_disp':700}}))
        return paths
    monkeypatch.setattr(runner,'generate_mock',generate)
    monkeypatch.setattr(runner,'solver_config_for_mock',lambda *a,**k:None)
    monkeypatch.setattr(runner,'_fisher_for_seed',lambda *a,**k:None)
    calls=[]
    monkeypatch.setattr(runner.pipeline,'run',lambda c:calls.append('sample'))
    def recovery(directory,*a,**k):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/runner.RECOVERY_FILE).write_text('{}')
        calls.append('saved')
    monkeypatch.setattr(runner,'write_recovery',recovery)
    def plots(root):
        calls.append(('plots',len(list(root.glob('seed_*/recovery/recovery.json')))))
        return []
    monkeypatch.setattr(aggregate_plots,'write_aggregate_plots',plots)
    runner.run_mock_validation(config,log=lambda m:None)
    assert calls==['sample','saved',('plots',1),'sample','saved',('plots',2)]
    calls.clear()
    runner.run_mock_validation(config,log=lambda m:None)
    assert calls==[('plots',2)]

def test_plot_failure_logged(tmp_path,monkeypatch):
    def fail(root):raise RuntimeError('plot failure')
    monkeypatch.setattr(aggregate_plots,'write_aggregate_plots',fail)
    logs=[]
    runner._refresh_aggregate_plots(tmp_path,log=logs.append)
    assert 'FAILED: RuntimeError: plot failure' in logs[-1]


def test_refresh_reports_failed_convergence_without_excluding(tmp_path, monkeypatch):
    def plots(root):
        output=root/'aggregate_plots'
        output.mkdir()
        (output/'campaign_selection.json').write_text(json.dumps(dict(
            included=2, realizations=2, converged=1, rule='test cut', all_realizations_used=True,
            failed_convergence=[dict(seed=10,reasons='max R-hat 1.08')],excluded=[])))
        return []
    monkeypatch.setattr(aggregate_plots,'write_aggregate_plots',plots)
    logs=[]
    runner._refresh_aggregate_plots(tmp_path, log=logs.append)
    assert any('included 2/2' in line for line in logs)
    assert any('included despite failed convergence seed=10' in line for line in logs)

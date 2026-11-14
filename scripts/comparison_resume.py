"""Durable, content-checked work units for the Colab comparison (stdlib only)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def copy_verified(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected['size'] and file_hash(target) == expected['sha256']:
        return
    tmp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        shutil.copyfile(source, tmp)
        if tmp.stat().st_size != expected['size'] or file_hash(tmp) != expected['sha256']:
            raise IOError(f'Incomplete copy: {target}')
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def select_run(local_parent, drive_parent, identity, requested='latest'):
    """Choose the newest compatible run, or create one; explicit mismatches fail."""
    parents = [Path(p) for p in (local_parent, drive_parent) if p is not None]
    if requested == 'latest':
        matches = set()
        for parent in parents:
            for manifest in parent.glob('*/resume_manifest.json'):
                data = read_json(manifest)
                if data and data.get('identity') == identity:
                    matches.add(manifest.parent.name)
        requested = max(matches) if matches else 'new'
    if requested == 'new':
        requested = time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:6]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', requested):
        raise ValueError('Run ID must contain only letters, digits, underscores and hyphens')
    for parent in parents:
        path = parent/requested
        if path.exists():
            old = read_json(path/'resume_manifest.json')
            if old is None or old.get('identity') != identity:
                raise ValueError(f'Cannot safely resume {requested}: settings/source/checkpoints differ, '
                                 'or this run predates completion receipts. Use latest or new.')
    manifest = dict(version=1, run_id=requested, identity=identity)
    for parent in parents:
        atomic_json(parent/requested/'resume_manifest.json', manifest)
    return requested


class ResumeCache:
    def __init__(self, root, drive_root=None, repo=None):
        self.root = Path(root)
        self.drive_root = Path(drive_root) if drive_root else None
        self.repo = Path(repo) if repo else self.root
        manifest = read_json(self.root/'resume_manifest.json')
        if not manifest:
            raise ValueError('Missing resume manifest; run the resume setup cell first')
        self.context = digest(manifest['identity'])
        self._checked = {}

    def key(self, kind, output):
        return str(Path(output).relative_to(self.root)) + ':' + kind

    def receipt_path(self, key, root=None):
        return (Path(root) if root else self.root)/'.resume'/(digest(key)+'.json')

    def signature(self, command, dependencies=()):
        normalized = [str(c).replace(str(self.root), '$RUN').replace(str(self.repo), '$REPO') for c in command]
        return digest(dict(context=self.context, command=normalized, dependencies=list(dependencies)))

    def _matches(self, path, expected):
        if not path.is_file(): return False
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, expected['sha256'])
        if key not in self._checked:
            self._checked[key] = stat.st_size == expected['size'] and file_hash(path) == expected['sha256']
        return self._checked[key]

    def _valid(self, receipt, root):
        if not receipt or receipt.get('context') != self.context or not receipt.get('files'):
            return False
        for dep in receipt.get('dependencies', []):
            upstream = read_json(self.receipt_path(dep['key'], root))
            if dep['hash'] is None:
                if upstream and self._valid(upstream, root): return False
            elif digest(upstream) != dep['hash'] or not self._valid(upstream, root):
                return False
        for name, expected in receipt['files'].items():
            path = Path(root)/name
            if not self._matches(path, expected):
                return False
        return True

    def restore(self, key, signature=None):
        for root in [self.root] + ([self.drive_root] if self.drive_root else []):
            receipt = read_json(self.receipt_path(key, root))
            if receipt and receipt.get('key') == key and (signature is None or receipt.get('signature') == signature) and self._valid(receipt, root):
                if root != self.root:
                    print('Restoring from Drive:', key, flush=True)
                    for name, expected in receipt['files'].items():
                        copy_verified(root/name, self.root/name, expected)
                    atomic_json(self.receipt_path(key), receipt)
                # A previous interrupted Drive copy is retried before skipping work.
                self.publish(receipt)
                return receipt
        return None

    def publish(self, receipt):
        if self.drive_root is None:
            return
        remote = self.receipt_path(receipt['key'], self.drive_root)
        if read_json(remote) == receipt and self._valid(receipt, self.drive_root):
            return
        remote.unlink(missing_ok=True)
        for name, expected in receipt['files'].items():
            copy_verified(self.root/name, self.drive_root/name, expected)
        atomic_json(remote, receipt)  # Commit completion only after verified copies.

    def invalidate(self, key):
        self.receipt_path(key).unlink(missing_ok=True)
        if self.drive_root:
            self.receipt_path(key, self.drive_root).unlink(missing_ok=True)

    def complete(self, key, signature, files, dependencies=()):
        records = {}
        for path in sorted(set(map(Path, files))):
            if not path.is_file():
                raise FileNotFoundError(path)
            records[str(path.relative_to(self.root))] = dict(size=path.stat().st_size, sha256=file_hash(path))
        if not records:
            raise ValueError(f'No output artifacts for {key}')
        receipt = dict(version=1, key=key, context=self.context, signature=signature, files=records, dependencies=list(dependencies))
        atomic_json(self.receipt_path(key), receipt)
        print('Saving completed data:', key, flush=True)
        self.publish(receipt)
        print('Saved completed unit:', key, flush=True)
        return receipt

    def token(self, kind, output, *, optional=False):
        key = self.key(kind, output)
        receipt = self.restore(key)
        if receipt is None and not optional:
            raise RuntimeError(f'Upstream output is incomplete: {key}; rerun its cell first')
        return dict(key=key, hash=digest(receipt) if receipt else None)

    def completed_outputs(self, kind, arm):
        keys = set()
        for root in [self.root] + ([self.drive_root] if self.drive_root else []):
            for path in (root/'.resume').glob('*.json'):
                row = read_json(path)
                if row and row.get('key', '').startswith(arm+'/') and row['key'].endswith(':'+kind):
                    keys.add(row['key'])
        return [self.root/key.rsplit(':', 1)[0] for key in sorted(keys) if self.restore(key)]

    def run(self, command, log_path, *, env=None):
        """Cache each supported command separately, with upstream dependencies."""
        command = list(map(str, command))
        kind, output, required, patterns, dependencies = self.plan(command)
        key = self.key(kind, output)
        signature = self.signature(command, dependencies)
        if self.restore(key, signature):
            print('SKIP completed:', key, flush=True)
            return 0
        self.invalidate(key)
        # Remove only artifacts owned by this unit. Partial files cannot satisfy a retry.
        for pattern in patterns:
            for path in output.glob(pattern):
                if path.is_file(): path.unlink()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print('RUN:', key, flush=True)
        with log_path.open('w') as log:
            with subprocess.Popen(command, cwd=self.repo, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                try:
                    for line in proc.stdout:
                        print(line, end='', flush=True)
                        log.write(line); log.flush()
                    status = proc.wait()
                except BaseException:
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    raise
        if status:
            print(f'FAILED (will retry): {key}; exit {status}', flush=True)
            return status
        for name in required:
            path = output/name
            if not path.is_file(): raise FileNotFoundError(f'Success exit but missing output: {path}')
            if path.suffix == '.json' and read_json(path) is None:
                raise ValueError(f'Invalid JSON output: {path}')
        files = [p for pattern in patterns for p in output.glob(pattern) if p.is_file()]
        self.complete(key, signature, files, dependencies)
        return 0

    def save_aux(self, path, required=True):
        if self.drive_root is None: return
        names = {'stage_status.json', 'manifest.json', 'fidelity.json', 'comparison.json',
                 'comparison.csv', 'source_snapshot.json', 'source_manifest.json'}
        try:
            for source in Path(path).rglob('*'):
                if source.is_file() and (source.name in names or source.suffix == '.log'):
                    target = self.drive_root/source.relative_to(self.root)
                    copy_verified(source, target, dict(size=source.stat().st_size, sha256=file_hash(source)))
        except Exception:
            if required: raise
            print('Could not save auxiliary logs to Drive; completed data receipts are separate.', flush=True)

    def plan(self, command):
        index = next(i for i, c in enumerate(command) if c.endswith('.py'))
        script, args = Path(command[index]).name, command[index+1:]
        def arg(flag): return Path(args[args.index(flag)+1])
        def token(kind, path, optional=False): return self.token(kind, path, optional=optional)
        deps = []
        if script == 'probe_pi05_transcoder_counterfactual.py':
            kind, output = 'probe', arg('--output-dir')
            required = ['decision_metrics.json', 'counterfactual_summary.json', 'provenance.json', 'latent_deltas.csv', 'latents/index.json']
            patterns = required + ['observation_shapes.json', 'scene_objects.json', 'latents/*.npz', 'images/*.png', '*_deltas.csv', 'measurements.csv', 'h1_cells.csv']
        elif script == 'report_pi05_counterfactual_features.py':
            kind, output = 'nomination', Path(args[0])
            required = patterns = ['nominated_targets.json', 'candidate_features.csv', 'layer_selectivity.csv']
            deps = [token('probe', output)]
        elif script == 'collect_pi05_transcoder_features.py':
            kind, output = 'discovery', arg('--output-dir')
            required = ['config.json', 'feature_stats.pt', 'feature_topk.pt', 'observations.jsonl']
            patterns = required + ['collection_metrics.jsonl', 'token_sparsity.pt']
        elif script == 'trace_pi05_transcoder_circuit.py':
            kind, output = 'trace', arg('--output-dir')
            required = ['graph.json', 'trace_config.json']
            patterns = required + ['nodes_summary.csv', 'edges.csv', 'edges_summary.csv']
            deps = [token('discovery', arg('--feature-dir'))]
        elif script == 'calibrate_circuit_audit.py':
            kind, output = 'calibration', arg('--output-dir')
            required = ['audit_calibration.json']
            patterns = required + ['audit_calibration.csv']
            deps = [token('probe', Path(args[0]))]
        elif script == 'validate_circuit_with_counterfactual.py':
            kind, output = 'audit', Path(args[0])/'counterfactual_validation'
            required = ['validation.json']
            patterns = required + ['circuit_node_responses.csv']
            deps = [token('trace', Path(args[0])), token('probe', Path(args[1]))]
        elif script == 'aggregate_counterfactual_runs.py':
            kind, output = 'aggregate', arg('--output-dir')
            required = patterns = ['aggregate.json', 'per_task.csv', 'h2_per_task.csv']
            deps = [token('probe', Path(p)) for p in args[:args.index('--output-dir')]]
        elif script == 'summarize_h3_rate.py':
            kind, output = 'summary', arg('--output-dir')
            required = patterns = ['h3_rate.json', 'h3_rate.csv']
            first_flag = next((i for i, a in enumerate(args) if a.startswith('--')), len(args))
            deps = []
            for p in args[:first_flag]:
                audit_dir = Path(p)/'counterfactual_validation'
                dep = token('audit', audit_dir, optional=True)
                if dep['hash'] is None:
                    (audit_dir/'validation.json').unlink(missing_ok=True)
                deps.append(dep)
            deps += [token('calibration', Path(args[i+1])) for i, a in enumerate(args) if a == '--calibration']
        else:
            raise ValueError(f'Unsupported resumable command: {script}')
        return kind, output, required, patterns, deps

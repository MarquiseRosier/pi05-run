"""Dependency preflight regressions; runnable without GPU/model dependencies."""
import ast
import importlib.metadata
import json
import io
from pathlib import Path
import sys
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pi05_monitor_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setup_source(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                              'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
        return next(''.join(c['source']) for c in notebook['cells']
                    if ''.join(c.get('source', [])).startswith('# @title Install Pinned'))

    def test_setup_streams_installer_error_and_keeps_it_in_exception(self):
        tree = ast.parse(self.setup_source())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        scope = {'subprocess': subprocess}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), '<setup>', 'exec'), scope)
        output = io.StringIO()
        process = Mock()
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        def lines():
            yield 'Resolving packages\n'
            self.assertIn('Resolving packages', output.getvalue())
            yield 'error: failed to build dependency wheel\n'
        process.stdout = lines()
        process.wait.return_value = 1
        with patch.object(subprocess, 'Popen', return_value=process) as popen, redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, 'failed to build dependency wheel'):
                scope['run'](['uv', 'pip', 'install', 'example'], env={'SETUP': '1'})
        self.assertEqual(popen.call_args.kwargs['stderr'], subprocess.STDOUT)
        self.assertEqual(popen.call_args.kwargs['env'], {'SETUP': '1'})
        self.assertIn('error: failed to build dependency wheel', output.getvalue())

    def test_setup_system_install_explicitly_targets_the_active_kernel(self):
        source = self.setup_source()
        tree = ast.parse(source)
        command = next(n for n in ast.walk(tree) if isinstance(n, ast.List)
                       and any(isinstance(v, ast.Constant) and v.value == '--torch-backend' for v in n.elts))
        arguments = eval(compile(ast.Expression(command), '<install args>', 'eval'), {
            'UV': '/content/uv-bin/uv', 'PYTHON': Path('/usr/bin/python3'),
            'transformers_override': Path('/content/pi05-transformers-overrides.txt')})
        self.assertEqual(arguments[arguments.index('--python') + 1], '/usr/bin/python3')
        self.assertIn('--system', arguments)
        self.assertIn('--break-system-packages', arguments)
        self.assertIn('--overrides', arguments)
        self.assertIn(runtime.CMAKE_REQUIREMENT, arguments)
        self.assertEqual(arguments[arguments.index('--no-build-isolation-package') + 1], 'egl-probe')
        build = source.index('"cmake==3.31.6", "setuptools>=71,<81", "wheel"')
        egl = source.index('"egl-probe==1.0.2"')
        policy = source.index('"lerobot[libero]==0.4.4"')
        self.assertLess(build, egl)
        self.assertLess(egl, policy)

    def siglip(self, result=True):
        module = ModuleType('transformers.models.siglip')
        module.check = SimpleNamespace(
            check_whether_transformers_replace_is_installed_correctly=lambda: result)
        return module

    def test_incompatible_transformers_fail_before_lerobot_imports(self):
        no_check_function = self.siglip()
        no_check_function.check = SimpleNamespace()
        for siglip in (None, ModuleType('transformers.models.siglip'),
                       no_check_function, self.siglip(False)):
            with self.subTest(siglip=siglip), patch.dict(sys.modules, {
                'transformers.models.siglip': siglip,
                'lerobot.envs.configs': None,
            }), patch.object(importlib.metadata, 'version', return_value='0.4.4'):
                with self.assertRaisesRegex(RuntimeError, r'pinned.*Transformers.*--repair'):
                    runtime.validate_runtime()

    def test_compatible_fork_passes_without_loading_weights(self):
        with patch.dict(sys.modules, {'transformers.models.siglip': self.siglip()}):
            runtime.validate_pi_transformers()

    def test_wrong_lerobot_is_rejected_first(self):
        with patch.object(importlib.metadata, 'version', return_value='0.6.1'), \
             patch.object(runtime, 'validate_pi_transformers') as transformer_check:
            with self.assertRaisesRegex(RuntimeError, 'requires lerobot==0.4.4'):
                runtime.validate_runtime()
            transformer_check.assert_not_called()

    def test_repair_overrides_pypi_transformers_and_checks_in_target_python(self):
        args = ['pi05_monitor_runtime.py', '--repair', '--uv', '/test/uv',
                '--python', '/test/python', '--require-cuda']
        with patch.object(sys, 'argv', args), patch.object(runtime.subprocess, 'run') as run:
            runtime.main()
        build_tools, install, check = run.call_args_list
        self.assertIn(runtime.CMAKE_REQUIREMENT, build_tools.args[0])
        self.assertIn('wheel', build_tools.args[0])
        self.assertIn(runtime.CMAKE_REQUIREMENT, install.args[0])
        self.assertEqual(install.args[0][install.args[0].index('--no-build-isolation-package') + 1], 'egl-probe')
        self.assertIn('lerobot[libero]==0.4.4', install.args[0])
        self.assertNotIn('--no-deps', install.args[0])
        self.assertEqual(install.args[0][install.args[0].index('--overrides') + 1],
                         str(runtime.TRANSFORMERS_OVERRIDES))
        self.assertIn('--reinstall-package', install.args[0])
        self.assertEqual(install.args[0][install.args[0].index('--reinstall-package') + 1],
                         'transformers')
        self.assertEqual(install.args[0][:5],
                         ['/test/uv', 'pip', 'install', '--python', '/test/python'])
        self.assertEqual(check.args[0][0], '/test/python')
        self.assertIn('--check-here', check.args[0])
        self.assertIn('--require-cuda', check.args[0])
        self.assertTrue(install.kwargs['check'])
        self.assertTrue(check.kwargs['check'])

    def test_ensure_healthy_environment_does_not_install(self):
        with patch.object(sys, 'argv', ['runtime', '--ensure', '--python', '/test/python']), \
             patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            runtime.main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], '/test/python')
        self.assertIn('--check-here', run.call_args.args[0])

    def test_ensure_repairs_then_rechecks_in_fresh_process(self):
        results = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0),
                   SimpleNamespace(returncode=0), SimpleNamespace(returncode=0)]
        args = ['runtime', '--ensure', '--python', '/test/python', '--uv', '/test/uv', '--require-cuda']
        with patch.object(sys, 'argv', args), \
             patch.object(runtime.subprocess, 'run', side_effect=results) as run:
            runtime.main()
        probe, build_tools, install, recheck = run.call_args_list
        self.assertIn(runtime.CMAKE_REQUIREMENT, build_tools.args[0])
        self.assertEqual(probe.args[0], recheck.args[0])
        self.assertIn('--require-cuda', recheck.args[0])
        self.assertIn('lerobot[libero]==0.4.4', install.args[0])
        self.assertFalse(probe.kwargs['check'])
        self.assertTrue(recheck.kwargs['check'])

    def test_failed_repair_or_recheck_stops_ensure(self):
        for failure_index in (1, 2, 3):
            results = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0),
                       SimpleNamespace(returncode=0), SimpleNamespace(returncode=0)]
            results[failure_index] = subprocess.CalledProcessError(1, ['failed-command'])
            with self.subTest(failure_index=failure_index), \
                 patch.object(sys, 'argv', ['runtime', '--ensure']), \
                 patch.object(runtime.subprocess, 'run', side_effect=results) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime.main()
                self.assertEqual(run.call_count, failure_index + 1)

    def test_notebook_runs_in_kernel_and_propagates_original_error(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                              'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if ''.join(c.get('source', [])).startswith('# @title Run Budgeted'))
        display = ModuleType('IPython.display')
        display.display = display.Markdown = display.Image = lambda *args, **kwargs: None
        error = ValueError('original model error')
        runner = Mock(side_effect=error)
        validator = Mock(return_value={'status': 'passed'})
        before = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, patch.object(sys, 'path', sys.path.copy()), \
             patch.dict(sys.modules, {'IPython.display': display,
                                      'pi05_monitor_runtime': SimpleNamespace(validate_runtime=validator),
                                      'run_pi05_monitor_budget': SimpleNamespace(main=runner)}), \
             patch.object(subprocess, 'run', side_effect=AssertionError('no child Python')):
            scope = dict(RUN_NAME='test-run', REQUIRE_MATCHED_COMPARISON=False, CHECKPOINT='',
                         DRIVE_OUTPUTS=Path(folder), LOCAL_REPO=Path(folder), BUDGET_MINUTES=15,
                         EFFECTIVE_BATCH_SIZE=4, PRESET='a100')
            with self.assertRaises(ValueError) as raised:
                exec(source, scope)
            self.assertIs(raised.exception, error)
            runner.assert_called_once()
            self.assertIn('--batch-size', runner.call_args.args[0])
            self.assertEqual(Path.cwd(), before)

    def test_notebook_reuse_rejects_stock_transformers(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                              'notebooks/pi05_occlusion_monitor_pilot.ipynb').read_text())
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        for source in sources:
            ast.parse(source)
        install = next(s for s in sources if s.startswith('# @title Install Pinned'))
        self.assertIn(runtime.REQUIREMENT, install)
        pinned_fork = next(line for line in runtime.TRANSFORMERS_OVERRIDES.read_text().splitlines()
                           if line.startswith('transformers @'))
        self.assertIn(pinned_fork, install)
        self.assertIn('"--overrides", str(transformers_override)', install)
        tree = ast.parse(install)
        self.assertIn('PYTHON = Path(sys.executable)', install)
        probe = next(n for n in tree.body if isinstance(n, ast.Try))
        command = compile(ast.Module(body=[probe], type_ignores=[]), '<readiness>', 'exec')
        for siglip in (None, self.siglip(False), self.siglip(True)):
            with self.subTest(siglip=siglip), patch.dict(sys.modules, {
                'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                'transformers.models.siglip': siglip,
            }), patch.object(importlib.metadata, 'version', return_value='0.4.4'), \
                 patch('importlib.util.find_spec', return_value=True):
                scope = {}
                exec(command, scope)
                self.assertEqual(scope['ready'], siglip is not None and
                                 siglip.check.check_whether_transformers_replace_is_installed_correctly())


if __name__ == '__main__':
    unittest.main()

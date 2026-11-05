"""Early diagnosis of the actual old-Conda-stack failure, without GPU imports."""
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from ScopeSFT import preflight


class TrainingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("transformers/masking_utils.py", "transformers/models/qwen3_5/modeling_qwen3_5.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.distribution = Mock()
        self.distribution.locate_file.side_effect = lambda name: self.root / name
        self.versions = dict(preflight.REQUIRED_VERSIONS)

    def checks(self, **kwargs):
        with patch.object(preflight.metadata, "version", side_effect=self.versions.__getitem__), \
                patch.object(preflight.metadata, "distribution", return_value=self.distribution):
            return preflight.check_training_environment(**kwargs)

    def test_verified_cpu_and_cuda_builds_report_interpreter_and_versions(self):
        for torch_version in ("2.5.1", "2.5.1+cpu", "2.5.1+cu118"):
            with self.subTest(torch_version=torch_version), patch.dict("os.environ", {"CONDA_PREFIX": "/conda/test-env"}):
                self.versions["torch"] = torch_version
                report = self.checks()
                self.assertEqual(report["python_executable"], sys.executable)
                self.assertEqual(report["conda_prefix"], "/conda/test-env")
                self.assertEqual(report["versions"]["torch"], torch_version)
                self.assertIn("transformers/masking_utils.py", report["checked_sources"])

    def test_old_transformers_is_diagnosed_before_any_model_source_inspection(self):
        self.versions["transformers"] = "4.51.3"
        with self.assertRaises(preflight.TrainingEnvironmentError) as raised:
            self.checks()
        message = str(raised.exception)
        self.assertIn("transformers=4.51.3", message)
        self.assertIn("transformers=5.3.0", message)
        self.assertIn(sys.executable, message)
        self.assertIn("conda activate scope-qwen35", message)
        self.assertIn("-m pip install -r", message)
        self.assertIn("不是 CUDA 显存不足", message)
        self.distribution.locate_file.assert_not_called()

    def test_all_incompatible_core_versions_are_reported_together(self):
        self.versions.update(torch="2.6.0+cu124", peft="0.14.0", accelerate="1.6.0")
        with self.assertRaises(preflight.TrainingEnvironmentError) as raised:
            self.checks()
        for actual in ("2.6.0+cu124", "0.14.0", "1.6.0"):
            self.assertIn(actual, str(raised.exception))

    def test_missing_package_has_actionable_error_instead_of_import_traceback(self):
        def version(name):
            if name == "transformers":
                raise metadata.PackageNotFoundError(name)
            return self.versions[name]
        with patch.object(preflight.metadata, "version", side_effect=version), \
                self.assertRaises(preflight.TrainingEnvironmentError) as raised:
            preflight.check_training_environment()
        self.assertIn("transformers=未安装", str(raised.exception))
        self.assertIn("requirements.txt", str(raised.exception))

    def test_masking_utils_missing_with_correct_metadata_reports_broken_install(self):
        (self.root / "transformers/masking_utils.py").unlink()
        with self.assertRaises(preflight.TrainingEnvironmentError) as raised:
            self.checks()
        self.assertIn("安装不完整", str(raised.exception))
        self.assertIn("transformers/masking_utils.py", str(raised.exception))

    def test_native_model_file_is_required_only_for_model_preflight(self):
        (self.root / "transformers/models/qwen3_5/modeling_qwen3_5.py").unlink()
        report = self.checks(require_model=False)
        self.assertEqual(report["checked_sources"], ["transformers/masking_utils.py"])
        with self.assertRaisesRegex(preflight.TrainingEnvironmentError, "modeling_qwen3_5.py"):
            self.checks()

    def test_fresh_process_old_stack_failure_does_not_import_torch_or_transformers(self):
        program = '''
import json, sys
from unittest.mock import patch
from ScopeSFT import preflight
versions = dict(preflight.REQUIRED_VERSIONS, transformers="4.51.3")
with patch.object(preflight.metadata, "version", side_effect=versions.__getitem__):
    try:
        preflight.check_training_environment()
    except preflight.TrainingEnvironmentError as error:
        print(json.dumps({"message": str(error), "torch_imported": "torch" in sys.modules,
                          "transformers_imported": "transformers" in sys.modules}))
    else:
        raise SystemExit("Expected old stack to fail")
'''
        completed = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=20, check=True)
        report = json.loads(completed.stdout)
        self.assertFalse(report["torch_imported"])
        self.assertFalse(report["transformers_imported"])
        self.assertIn("4.51.3", report["message"])


if __name__ == "__main__":
    unittest.main()

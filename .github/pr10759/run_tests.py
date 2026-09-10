# SPDX-License-Identifier: AGPL-3.0-only
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'evidence-output'
out.mkdir(exist_ok=True)
tests = sorted((root / 'studio/backend/tests').glob('test_research*.py'))
tests += sorted((root / 'studio/backend/tests').glob('test_deep_research*.py'))
env = dict(os.environ, PYTHONPATH=str(root / 'studio/backend'), COVERAGE_FILE=str(out / '.coverage'))
print(f'Running {len(tests)} research test modules', flush=True)
raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', *map(str, tests), '-q', '--timeout=90', '--cov=core.research_runs', '--cov=storage.research_runs_db', '--cov-branch', '--cov-report=term', f'--cov-report=json:{out}/coverage.json', f'--junitxml={out}/backend.xml'], cwd=root, env=env))

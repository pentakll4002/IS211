import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

def run(script):
    print(f'\n{"=" * 60}')
    print(f'Running: {script}')
    print(f'{"=" * 60}')
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, script)],
        cwd=BASE
    )
    if result.returncode != 0:
        print(f'FAILED: {script}')
        sys.exit(1)

if __name__ == '__main__':
    run('pipeline_01_eda.py')
    run('pipeline_02_training.py')
    run('pipeline_03_fragmentation.py')
    print('\nAll pipelines completed!')


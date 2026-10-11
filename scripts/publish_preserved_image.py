"""CPU-only Actions publication of an immutable public HF Docker archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def main():
    release = json.loads(Path('release-source.json').read_text())
    subprocess.run(['docker', 'system', 'prune', '-af'], check=True)
    # This is an ephemeral GitHub runner, not the research server.
    free = int(subprocess.check_output(['df', '--output=avail', '-B1', '/'], text=True).splitlines()[-1])
    assert free > release['image_bytes'] + 8_000_000_000, 'Runner needs sufficient image storage'
    decompressor = subprocess.Popen(['zstd', '-dc'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    loader = subprocess.Popen(['docker', 'load'], stdin=decompressor.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decompressor.stdout.close()
    digest = hashlib.sha256()
    count = 0
    try:
        with urllib.request.urlopen(release['archive_url'], timeout=600) as response:
            while block := response.read(8 * 1024 * 1024):
                digest.update(block)
                count += len(block)
                decompressor.stdin.write(block)
        decompressor.stdin.close()
        output = loader.communicate(timeout=600)[0].decode()
        assert decompressor.wait(timeout=600) == loader.returncode == 0, output
    finally:
        if loader.poll() is None: loader.kill()
        if decompressor.poll() is None: decompressor.kill()
    assert digest.hexdigest() == release['archive_sha256'] and count == release['archive_bytes']
    actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', release['image_id']], text=True))[0]
    assert actual['Id'] == release['image_id']
    subprocess.run(['docker', 'run', '--rm', '--entrypoint', '/opt/python/bin/python3.11',
        actual['Id'], '-c', "import json; from pathlib import Path; from submission.scope import validate_final_manifest; "
        "m=json.loads(Path('/models/model-manifest.json').read_text()); validate_final_manifest(m); "
        "assert set(m['routes'])=={'0','1','2','3','5'}; print('EMBEDDED_MANIFEST_VERIFIED')"], check=True)
    uri = release['registry_uri']
    subprocess.run(['docker', 'tag', actual['Id'], uri], check=True)
    subprocess.run(['docker', 'push', uri], check=True)
    subprocess.run(['docker', 'pull', uri], check=True)
    remote = json.loads(subprocess.check_output(['docker', 'image', 'inspect', uri], text=True))[0]
    assert remote['Id'] == actual['Id']
    repo_digest = next(d for d in remote['RepoDigests'] if d.startswith(uri.split(':')[0] + '@'))
    subprocess.run(['docker', 'pull', repo_digest], check=True)
    evidence = {'status': 'registry_pushed_authenticated_digest_pull_verified',
        'registry_uri': uri, 'registry_digest_uri': repo_digest, 'image_id': actual['Id'],
        'archive_sha256': digest.hexdigest(), 'archive_bytes': count,
        'github_run_id': os.environ['GITHUB_RUN_ID'], 'public_anonymous_pull_verified': False,
        'physical_rollouts': 0, 'actual_24gb_tested': False}
    Path('registry-release.json').write_text(json.dumps(evidence, indent=2) + '\n')
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write('## Registry publication\n\n```json\n' + json.dumps(evidence, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()

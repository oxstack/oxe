"""Replay the native boot command in private ESP/module/sys/lock fixtures.

No root, hardware, real /boot, real module tree, EFI variables or network.
The expected results come from hosted runs of the legacy Python command.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
RUST = str(Path(sys.argv[1]).resolve())
FIXTURE = '/tmp/oxa-memlat-fixture'
AUTO = 'Local AI memlat (newer-kernel)'
COMMENT = 'Auto-rebuilt Yoga MEMLAT; stock fallback Omarchy/linux-omarchy'
BASE = '''default_entry: Local AI memlat test (fixture)
/+Omarchy
  //linux-omarchy
  protocol: efi
  path: boot():/EFI/Linux/stock.efi
  cmdline: root=fixture quiet extra=outer
/Local AI memlat test (fixture)
protocol: efi
path: boot():/EFI/LocalAI/qualified.efi
'''
STOCK = b'\xd0\x0d\xfe\xedstock fixture'
CHANGED = b'\xd0\x0d\xfe\xedpatched fixture'

def uki():
    data = bytearray(2048)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<H', data, 0x86, 2)
    struct.pack_into('<H', data, 0x94, 240)
    struct.pack_into('<H', data, 0x98, 0x20b)
    for i, (name, payload) in enumerate(((b'.uname', b'fixture-kernel\0'), (b'.dtbauto', STOCK))):
        h = 0x98 + 240 + i * 40
        data[h:h+len(name)] = name
        struct.pack_into('<4I', data, h+8, len(payload), 4096*(i+1), 512, 1024+i*512)
        data[1024+i*512:1024+i*512+len(payload)] = payload
    return bytes(data)

def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def snapshot_digest(files):
    manifest = {name: digest(data) for name, data in files.items()}
    return digest(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode())

def execute(root, args):
    # The same boot-related roots are covered as in the reference capture.
    command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
               '--tmpfs', '/tmp', '--bind', str(root), FIXTURE,
               '--bind', str(root/'esp'), '/boot', '--ro-bind', str(root/'sys'), '/sys',
               '--ro-bind', str(root/'modules'), '/usr/lib/modules',
               '--bind', str(root/'locks'), '/run/lock', '--unshare-net',
               '--setenv', 'HOME', FIXTURE+'/home', '--setenv', 'PATH', FIXTURE+'/bin:/usr/bin:/bin']
    command += [RUST] + args + ([] if args in (['--help'], ['bogus'], ['enable','extra'], ['--policy']) else ['--policy', '/boot/policy.json'])
    return subprocess.run(command, capture_output=True, timeout=30)

def fixture(root, case):
    for name in ('esp/EFI/Linux', 'esp/EFI/LocalAI', 'modules/fixture-kernel/build', 'sys/firmware/efi/efivars', 'locks', 'bin', 'home'):
        (root/name).mkdir(parents=True, exist_ok=True)
    (root/'sys/firmware/efi/efivars/SecureBoot-fixture').write_bytes(b'\0'*5)
    (root/'modules/fixture-kernel/build/.config').write_text('CONFIG_SCMI_QCOM_MEMLAT_DEVFREQ=y\n')
    (root/'esp/EFI/Linux/stock.efi').write_bytes(uki())
    (root/'esp/EFI/LocalAI/qualified.efi').write_bytes(b'qualified memlat fixture')
    (root/'esp/stock.dtb').write_bytes(STOCK)
    (root/'esp/changed.dtb').write_bytes(CHANGED)
    (root/'esp/overlay.dtso').write_text('synthetic overlay passed to fake dtc\n')
    # Existing external DTC protocol, no alternate subject build path.
    (root/'bin/dtc').write_text('#!/bin/sh\nwhile [ "$1" != "-o" ]; do shift; done\nshift\nprintf fixture > "$1"\n')
    (root/'bin/fdtoverlay').write_text('#!/bin/sh\nwhile [ "$1" != "-o" ]; do shift; done\nshift\ncp /boot/changed.dtb "$1"\n')
    for tool in (root/'bin').iterdir():
        tool.chmod(0o755)
    policy = dict(config='/boot/limine.conf', stock_entry='Omarchy/linux-omarchy',
                  memlat_entry='Local AI memlat test (fixture)', stock_image='/boot/EFI/Linux/stock.efi',
                  stock_sha256=hashlib.sha256(uki()).hexdigest(), modules='/usr/lib/modules/fixture-kernel',
                  memlat_image='/boot/EFI/LocalAI/qualified.efi', memlat_sha256=hashlib.sha256(b'qualified memlat fixture').hexdigest(),
                  pending='/boot/pending', image_directory='/boot/EFI/LocalAI', boot_root='/boot',
                  builder=FIXTURE+'/oracle/memlat.py', overlay='/boot/overlay.dtso', stock_dtb='/boot/stock.dtb',
                  notification_user='fixture', notification_uid=999999)
    text = BASE
    if case in ('adopted', 'lookalike', 'outside', 'duplicate', 'rebuild-adopted'):
        block = f'\n/{AUTO}\ncomment: {COMMENT}\nprotocol: efi\npath: boot():/EFI/LocalAI/omarchy-memlat-newer-abc.efi\n'
        if case == 'lookalike': block = block.replace('Auto-rebuilt', 'Hand-made')
        if case == 'outside': block = block.replace('/EFI/LocalAI/omarchy-', '/elsewhere/omarchy-')
        text = text.replace('default_entry: '+policy['memlat_entry'], 'default_entry: '+AUTO) + block
        if case == 'duplicate': text += block
        if case == 'rebuild-adopted':
            policy['memlat_entry'] = 'Local AI memlat (old-kernel)'
            policy['automatic_entry_block'] = block.replace('newer', 'old')
    if case == 'modules':
        shutil.rmtree(root/'modules/fixture-kernel')
    if case == 'hash':
        policy['stock_sha256'] = '0'*64
    if case == 'bad-image':
        policy['memlat_sha256'] = '0'*64
    if case == 'build-failure':
        (root/'esp/stock.dtb').write_bytes(b'invalid DTB fixture')
    if case == 'secure-boot':
        (root/'sys/firmware/efi/efivars/SecureBoot-fixture').write_bytes(b'\0'*4+b'\1')
    if case == 'signed':
        data=bytearray(uki());struct.pack_into('<I',data,0x98+64,1)
        (root/'esp/EFI/Linux/stock.efi').write_bytes(data)
        policy['stock_sha256']=hashlib.sha256(data).hexdigest()
    if case == 'missing-driver':
        (root/'modules/fixture-kernel/build/.config').write_text('# CONFIG_SCMI_QCOM_MEMLAT_DEVFREQ is not set\n')
    if case == 'overflow-dtb':
        (root/'esp/changed.dtb').write_bytes(CHANGED+b'x'*1024)
    if case == 'user-default':
        text=text.replace('default_entry: '+policy['memlat_entry'], 'default_entry: Omarchy/linux-omarchy')
    if case == 'nested-lookalike':
        text=text.replace('default_entry: '+policy['memlat_entry'], 'default_entry: Omarchy/'+AUTO)
        text=text.replace('/Local AI memlat test',f'  //{AUTO}\n  comment: {COMMENT}\n  protocol: efi\n  path: boot():/EFI/LocalAI/omarchy-memlat-fake.efi\n/Local AI memlat test')
    if case == 'duplicate-cmdline':
        text=text.replace('  cmdline: root=fixture quiet extra=outer','  cmdline: one\n  cmdline: two')
    if case == 'numeric':
        text=text.replace('default_entry: '+policy['memlat_entry'],'default_entry: 3')
    if case == 'remember': text += '\nremember_last_entry: yes\n'
    if case == 'include': text += '\ninclude: elsewhere\n'
    if case == 'empty-cmdline': text = text.replace('cmdline: root=fixture quiet extra=outer', 'cmdline:')
    if case == 'crlf': text=text.replace('\n','\r\n')
    (root/'esp/limine.conf').write_text(text)
    (root/'esp/policy.json').write_text(json.dumps(policy))

SCENARIOS = [
    ('ordinary', [['enable'], ['enable'], ['prepare-update'], ['prepare-update'], ['fallback']]),
    ('ordinary', [['rebuild']]),
    ('ordinary', [['rebuild', '--dry-run']]),
    ('ordinary', [['rebuild', '--force'], ['prepare-update'], ['rebuild']]),
    ('adopted', [['check'], ['prepare-update']]),
    ('adopted', [['prepare-update'], ['rebuild']]),
    ('rebuild-adopted', [['prepare-update'], ['rebuild']]),
    ('lookalike', [['prepare-update'], ['check']]),
    ('outside', [['prepare-update'], ['fallback']]),
    ('duplicate', [['prepare-update'], ['rebuild', '--force']]),
    ('modules', [['enable'], ['check']]),
    ('hash', [['enable'], ['check']]),
    ('bad-image', [['enable'], ['check']]),
    ('build-failure', [['prepare-update'], ['rebuild']]),
    ('remember', [['check']]), ('include', [['prepare-update']]),
    ('empty-cmdline', [['rebuild', '--force']]),
    ('ordinary', [['enable', '--force']]),
    ('secure-boot', [['rebuild','--force']]),
    ('signed', [['rebuild','--force']]),
    ('missing-driver', [['rebuild','--force']]),
    ('overflow-dtb', [['rebuild','--force']]),
    ('user-default', [['prepare-update'], ['fallback'], ['check']]),
    ('nested-lookalike', [['prepare-update'], ['check']]),
    ('duplicate-cmdline', [['check']]),
    ('numeric', [['prepare-update'], ['rebuild']]),
    ('crlf', [['prepare-update'], ['rebuild']]),
    ('ordinary', [['--help'], ['bogus'], ['enable','extra'], ['--policy']]),
]
receipt = json.loads(Path(__file__).with_name('boot-oracle.json').read_text())
assert receipt['version'] == 1
assert len(receipt['steps']) == sum(len(actions) for _, actions in SCENARIOS)
steps = iter(receipt['steps'])
with tempfile.TemporaryDirectory(prefix='memlat-differential-') as directory:
    root = Path(directory)
    for case, actions in SCENARIOS:
        for p in list(root.iterdir()):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        fixture(root, case)
        for args in actions:
            expected = next(steps)
            before = snapshot_digest(snapshot(root/'esp'))
            assert (case, args, before) == (expected['case'], expected['args'], expected['before']), (case, args, before, expected)
            result = execute(root, args)
            assert result.returncode == expected['exit'], (case, args, result)
            assert digest(result.stderr) == expected['stderr'], (case, args, result.stderr)
            if expected['stdout'] is not None:
                stdout = result.stdout if args == ['--help'] else result.stdout.splitlines()[-1]
                assert digest(stdout) == expected['stdout'], (case, args, result.stdout)
            after = snapshot_digest(snapshot(root/'esp'))
            assert after == expected['after'], (case, args, after, expected['after'], result)
    print(f"{len(receipt['steps'])} captured actual-command fixture steps passed")

# Exercise the real release staging script against a source-layout fixture.
with tempfile.TemporaryDirectory(prefix='memlat-package-') as directory:
    root=Path(directory)/'source'
    for name in ('packaging/release', 'engine/target/release', 'oxa/target/release',
                 'oxc/tools', 'oxc/src/oxc/runtime', 'oxc/packaging/memlat', 'engine/tools/ci',
                 'engine/vendor/ggml-subset'):
        (root/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPO/'packaging/release/stage.sh',root/'packaging/release/stage.sh')
    for product,name in (('engine','oxe'),('engine','oxe-server'),('oxa','oxa')):
        (root/product/'target/release'/name).write_text('#!/bin/sh\nexit 0\n')
    shutil.copy2(RUST,root/'oxa/target/release/local-ai-memlat-boot')
    (root/'oxc/tools/memlat_boot.py').write_text('raise RuntimeError("legacy fixture must not ship")\n')
    for name in ('controller.py', 'oxc', 'studio'):
        (root/'oxc/src/oxc/runtime'/name).write_text('# synthetic legacy payload\n')
    templates = ('05-local-ai-memlat-fallback.hook', '20-local-ai-memlat-fallback',
                 '85-local-ai-memlat-rebuild', '99-z-local-ai-memlat-rebuild.hook')
    template_source = root/'oxc/packaging/memlat'
    for name in templates:
        shutil.copy2(REPO/'oxc/packaging/memlat'/name, template_source/name)
    # Independent exclusion control: retirement of the real unit must not break this fixture.
    (template_source/'oxc-memlat-shutdown.service').write_text('[Service]\nExecStart=/usr/bin/false\n')
    (root/'engine/tools/ci/crate_licenses.sh').write_text('#!/bin/sh\n# fixture collector; no external packages\nexit 0\n')
    (root/'engine/vendor/ggml-subset/LICENSE').write_text('synthetic license fixture\n')
    (root/'LICENSE').write_text('synthetic license fixture\n')
    destination=Path(directory)/'payload'
    subprocess.run(['/usr/bin/bash',str(root/'packaging/release/stage.sh'),str(destination)],check=True)
    installed=destination/'usr/bin/local-ai-memlat-boot'
    assert installed.read_bytes()==Path(RUST).read_bytes()
    assert installed.stat().st_mode & 0o777 == 0o755
    staged_templates = destination/'usr/share/ox-llm/memlat'
    assert {path.name for path in staged_templates.iterdir()} == set(templates)
    for name in templates:
        staged = staged_templates/name
        assert staged.read_bytes() == (template_source/name).read_bytes()
        assert staged.stat().st_mode & 0o777 == 0o644
    assert not list(destination.rglob('oxc-memlat-shutdown.service'))
    assert not list(destination.rglob('memlat_boot.py'))
    for legacy in ('usr/share/ox-llm/oxc', 'usr/bin/oxc', 'usr/bin/oxc-diagnostics'):
        path = destination / legacy
        assert not path.exists() and not path.is_symlink(), f'legacy payload staged: {legacy}'
    print('actual release stage preserves native helper and inert templates, excludes Python and service')

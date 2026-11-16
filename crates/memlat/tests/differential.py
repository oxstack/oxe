"""Run both real commands in the SAME private ESP/module/sys/lock fixtures.

No root, hardware, real /boot, real module tree, EFI variables or network.
Python is a test oracle only and never installed by the Rust package.
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
PYTHON = str(REPO / 'oxc/tools/memlat_boot.py')
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

def execute(root, implementation, args):
    # All host boot-related roots are covered before either subject starts.
    command = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
               '--tmpfs', '/tmp', '--bind', str(root), str(root),
               '--bind', str(root/'esp'), '/boot', '--ro-bind', str(root/'sys'), '/sys',
               '--ro-bind', str(root/'modules'), '/usr/lib/modules',
               '--bind', str(root/'locks'), '/run/lock', '--unshare-net',
               '--setenv', 'HOME', str(root/'home'), '--setenv', 'PATH', str(root/'bin')+':/usr/bin:/bin']
    implementation = [str(root/'oracle/local-ai-memlat-boot') if x == PYTHON else x for x in implementation]
    command += implementation + args + ([] if args in (['--help'], ['bogus'], ['enable','extra'], ['--policy']) else ['--policy', '/boot/policy.json'])
    return subprocess.run(command, capture_output=True, timeout=30)

def fixture(root, case):
    for name in ('esp/EFI/Linux', 'esp/EFI/LocalAI', 'modules/fixture-kernel/build', 'sys/firmware/efi/efivars', 'locks', 'bin', 'home', 'oracle'):
        (root/name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(PYTHON,root/'oracle/local-ai-memlat-boot')
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
                  builder=str(REPO/'oxc/tools/memlat.py'), overlay='/boot/overlay.dtso', stock_dtb='/boot/stock.dtb',
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
with tempfile.TemporaryDirectory(prefix='memlat-differential-') as directory:
    root=Path(directory)
    count=0
    for case, actions in SCENARIOS:
        for p in list(root.iterdir()):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        fixture(root, case)
        for args in actions:
            before=snapshot(root/'esp')
            py=execute(root, ['/usr/bin/python3', PYTHON], args)
            expected=snapshot(root/'esp')
            shutil.rmtree(root/'esp'); (root/'esp').mkdir()
            for name, data in before.items():
                p=root/'esp'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            rust=execute(root, [RUST], args)
            actual=snapshot(root/'esp')
            assert py.stderr == rust.stderr, (case,args,py.stderr,rust.stderr)
            expected_code = 2 if args in (['bogus'], ['enable','extra'], ['--policy']) else 1 if case in ('duplicate','remember','include','duplicate-cmdline') or (case in ('modules','hash','bad-image') and args[0]=='enable') or (case in ('build-failure','secure-boot','signed','missing-driver','overflow-dtb') and args[0]=='rebuild') or (args[0]=='enable' and '--force' in args) else 0
            assert py.returncode == expected_code, (case,args,expected_code,py.returncode,py.stdout,py.stderr)
            assert py.returncode==rust.returncode, (case,args,py.returncode,rust.returncode,py.stderr,rust.stderr)
            if args == ['--help']: assert py.stdout == rust.stdout, (py.stdout,rust.stdout)
            if py.returncode == 0:
                assert py.stdout.splitlines()[-1] == rust.stdout.splitlines()[-1], (case,args,py.stdout,rust.stdout)
            assert actual==expected, (case,args, {k for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k)},py.stdout,rust.stdout,py.stderr,rust.stderr)
            count+=1
    print(f'{count} actual-command differential steps passed')

# Exercise the real release staging script against a source-layout fixture.
with tempfile.TemporaryDirectory(prefix='memlat-package-') as directory:
    root=Path(directory)/'source'
    for name in ('packaging/release', 'engine/target/release', 'oxa/target/release',
                 'oxc/tools', 'engine/tools/ci', 'engine/vendor/ggml-subset'):
        (root/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPO/'packaging/release/stage.py',root/'packaging/release/stage.py')
    for product,name in (('engine','oxe'),('engine','oxe-server'),('oxa','oxa')):
        (root/product/'target/release'/name).write_text('#!/bin/sh\nexit 0\n')
    shutil.copy2(RUST,root/'oxa/target/release/local-ai-memlat-boot')
    (root/'oxc/tools/memlat_boot.py').write_text('raise RuntimeError("legacy fixture must not ship")\n')
    (root/'engine/tools/ci/crate_licenses.py').write_text('# fixture collector; no external packages\n')
    (root/'engine/vendor/ggml-subset/LICENSE').write_text('synthetic license fixture\n')
    (root/'LICENSE').write_text('synthetic license fixture\n')
    destination=Path(directory)/'payload'
    subprocess.run(['/usr/bin/python3','-I',str(root/'packaging/release/stage.py'),str(destination)],check=True)
    installed=destination/'usr/bin/local-ai-memlat-boot'
    assert installed.read_bytes()==Path(RUST).read_bytes()
    assert installed.stat().st_mode & 0o777 == 0o755
    assert not list(destination.rglob('memlat_boot.py'))
    print('actual release stage preserves native helper bytes/mode and excludes Python tool')

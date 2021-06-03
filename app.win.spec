# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['app.py'],
             pathex=['.'],
             binaries=[
             ],
             datas=[
                # ('extract/vox-adv-cpk.pth.tar', 'extract'),
                ('extract/vox-cpk.pth.tar', 'extract'),
                ('config/*', 'config'),
                ('source_image_inputs/*', 'source_image_inputs'),
                # These two paths will vary on your machine.
                # Update this file before running pyinstaller
                # gdown disabled since it doesn't package properly anyway
                # ('/home/adjective/.local/share/virtualenvs/real-time-example-of-first-order-motion-mo-LCLaGxN7/lib/python3.9/site-packages/gdown','gdown'),
                # ('/usr/lib64/python3.9/site-packages/PIL','PIL'),
             ],
             hiddenimports=[
                 'gdown',
                 'gdown.download'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[
                 'certifi',
                 'matplotlib',
                 'sklearn'
             ],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='first-order-motion-tk')
    
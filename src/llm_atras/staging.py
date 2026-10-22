"""Optional sequential staging of network snapshots onto local scratch storage."""
from pathlib import Path
import fcntl
import shutil


def stage_snapshot(source, root, model_id):
    source=Path(source).resolve();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if model_id.replace('/','__') in ['', '.', '..']:raise ValueError('Invalid model identifier')
    dest=root/model_id.replace('/','__')/source.name
    with (root/'.copy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (dest/'.complete').exists():return dest
        dest.mkdir(parents=True,exist_ok=True)
        for file in source.iterdir():
            if not file.is_file():continue
            target=dest/file.name
            if target.exists() and target.stat().st_size==file.stat().st_size:continue
            temporary=dest/(file.name+'.partial')
            shutil.copyfile(file,temporary)
            temporary.replace(target)
        (dest/'.complete').touch()
    return dest

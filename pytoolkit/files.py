import os, time, shutil

__all__ = ['listdir', 'dir', 'subdirs',
           'fileparts', 'stem', 'filelist_without_path', 'build_filename',
           'mkdir', 'mkdirs', 'gen_filelist', 'copytree']

listdir = os.listdir

def dir(path, ext=None, use_cache=False, case_sensitive=False):
    """ returns an ordered list of filenames found in `path`.
    """
    def _dir(path, ext):
        files = [os.path.join(path, x) for x in os.listdir(path)]
        files = [x for x in files if os.path.isfile(x) and (ext == None or os.path.splitext(x)[-1] == ext)]
        if not case_sensitive:
            files = [x.lower() for x in files]
        return sorted(files)
    if ext is '':
        ext = None
    cache = os.path.join(path, 'cache.list')
    if use_cache and os.path.exists(cache):
        with open(cache, 'r') as f:
            lines = [s.rstrip() for s in f.readlines()]
            lines = [os.path.join(path, line) for line in lines]
            if not case_sensitive:
                lines = [line.lower() for line in lines]
            return lines
    sorted_files = _dir(path, ext)
    if use_cache and not os.path.exists(cache):
        write_filelist_to_file(filelist_without_path(sorted_files), cache)

    return sorted_files

def subdirs(path):
    """ returns a list of subdirs found in `path`. 
    """
    fds = [os.path.join(path, x) for x in os.listdir(path)]
    fds = [x for x in fds if os.path.isdir(x)]
    return fds

def fileparts(filename):
    """ returns (path, filename, ext).
    Examples:
        In : filesparts('path/filename.ext')
        Out: ('path', 'filename', '.ext')

        In : filesparts('filename.ext')
        Out: ('', 'filename', '.ext')
    """
    path, filename = os.path.split(filename)
    filename, ext = os.path.splitext(filename)
    return path, filename, ext

def stem(filename):
    """ removes the path component from a filename.
    """
    _, name, ext = fileparts(filename)
    return name + ext

def filelist_without_path(filelist):
    """ removes the path component from a list of filenames.
    """
    temp = [fileparts(fname) for fname in filelist]
    paths, names, exts = zip(*temp)
    return [name + ext for name, ext in zip(names, exts)]

def build_filename(path, fname):
    return os.path.join(path, stem(fname))

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def write_filelist_to_file(filelist, fname_out):
    with open(fname_out, 'w') as f:
        filelist = [x + '\n' for x in filelist]
        f.writelines(filelist)

def gen_filelist(path):
    filelist = dir(path, case_sensitive=True)
    filelist = filelist_without_path(filelist)
    fname_out = os.path.join(path, 'filelist.txt')
    write_filelist_to_file(filelist, fname_out)

def copytree(src, dst, without_files=False):
    def ig_file(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    shutil.copytree(src, dst, ignore=ig_file if without_files else None)

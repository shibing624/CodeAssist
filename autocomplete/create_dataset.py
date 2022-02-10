# -*- coding: utf-8 -*-
"""
@author:https://github.com/labmlai
@description: Parse all files and write to a single file
refer: https://github.com/labmlai/python_autocomplete/blob/master/python_autocomplete/create_dataset.py
"""
import re
import string
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePath
from typing import List, NamedTuple, Set
from typing import Optional
import numpy as np
from loguru import logger

PRINTABLE = set(string.printable)


class PythonFile(NamedTuple):
    relative_path: str
    project: str
    path: Path


def get_python_files():
    """
    Get list of python files and their paths inside `data/source` folder
    """

    source_path = Path('download/source')
    files: List[PythonFile] = []

    def _add_file(path: Path):
        """
        Add a file to the list of tiles
        """
        project = path.relative_to(source_path).parents
        relative_path = path.relative_to(source_path / project[len(project) - 3])

        files.append(PythonFile(relative_path=str(relative_path),
                                project=str(project[len(project) - 2]),
                                path=path))

    def _collect_python_files(path: Path):
        """
        Recursively collect files
        """
        for p in path.iterdir():
            if p.is_dir():
                _collect_python_files(p)
            else:
                _add_file(p)

    _collect_python_files(source_path)

    return files


def read_file(path: Path) -> str:
    """
    Read a file
    """
    with open(str(path)) as f:
        content = f.read()

    content = ''.join(filter(lambda x: x in PRINTABLE, content))

    return content


def concat_and_save(path: PurePath, source_files: List[PythonFile]):
    with open(str(path), 'w') as f:
        for i, source in enumerate(source_files):
            f.write(f"# PROJECT: {source.project} FILE: {str(source.relative_path)}\n")
            f.write(read_file(source.path) + "\n")


def get_repos_from_readme(file_path='download/pytorch_awesome.md'):
    with open(str(file_path), 'r') as f:
        content = f.read()

    link_pattern = re.compile(r"""
        \[(?P<title>[^\]]*)\] # title
        \((?P<utl>[^\)]*)\) # url
    """, re.VERBOSE)

    res = link_pattern.findall(content)

    github_repos = []
    repo_pattern = re.compile(r'https://github.com/(?P<user>[^/]*)/(?P<repo>[^/#]*)$')
    for title, url in res:
        repos = repo_pattern.findall(url)
        for r in repos:
            github_repos.append((r[0], r[1]))

    return github_repos


def get_awesome_pytorch_readme(file_path='download/pytorch_awesome.md'):
    md = urllib.request.urlopen('https://raw.githubusercontent.com/bharathgs/Awesome-pytorch-list/master/README.md')
    content = md.read()

    with open(file_path, 'w', encoding='utf8') as f:
        f.write(str(content))


def download_repo(org: str, repo: str, idx: Optional[int]):
    zip_file = Path(f'download/{org}_{repo}.zip')

    if zip_file.exists():
        return zip_file

    if idx is not None:
        idx_str = f"{idx:03}: "
    else:
        idx_str = ""

    try:
        zip = urllib.request.urlopen(f'https://github.com/{org}/{repo}/archive/master.zip')
    except urllib.error.HTTPError as e:
        print(e)
        return
    content = zip.read()

    size = len(content) // 1024
    logger.debug(f"{idx_str} {org}/{repo} {size :,}KB")

    with open(str(zip_file), 'wb') as f:
        f.write(content)

    return zip_file


def create_folders():
    source = Path('download/source')
    if not source.exists():
        source.mkdir(parents=True)


def extract_zip(file_path: Path):
    source = Path('download/source')
    logger.debug(f"Extract {file_path}")
    repo_source = source / file_path.stem
    if repo_source.exists():
        return repo_source
    try:
        with zipfile.ZipFile(file_path, 'r') as repo_zip:
            repo_zip.extractall(repo_source)
    except zipfile.BadZipfile as e:
        print(file_path, e)

    return repo_source


def remove_files(path: Path, keep: Set[str]):
    """
    Remove files
    """
    for p in path.iterdir():
        if p.is_symlink():
            p.unlink()
            continue
        if p.is_dir():
            remove_files(p, keep)
        else:
            if p.suffix not in keep:
                p.unlink()


def progressive(limit_size=None):
    logger.debug('Get pytorch_awesome')
    # Get repos
    get_awesome_pytorch_readme()
    repos = get_repos_from_readme()
    if limit_size:
        repos = repos[:limit_size]
    # Download zips
    for i, r in enumerate(repos):
        zip_file = download_repo(r[0], r[1], i)
        if not zip_file:
            continue
        extracted = extract_zip(zip_file)
        remove_files(extracted, {'.py'})


def main():
    create_folders()
    try:
        progressive(limit_size=10)
    except KeyboardInterrupt:
        pass
    source_files = get_python_files()
    np.random.shuffle(source_files)
    logger.debug(f'Source_files size: {len(source_files)}')

    train_valid_split = int(len(source_files) * 0.9)
    train_file = 'download/train.txt'
    valid_file = 'download/valid.txt'
    concat_and_save(Path(train_file), source_files[:train_valid_split])
    concat_and_save(Path(valid_file), source_files[train_valid_split:])
    logger.info(f'Save train file: {train_file}, valid file: {valid_file}')


if __name__ == '__main__':
    main()

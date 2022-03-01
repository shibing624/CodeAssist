# -*- coding: utf-8 -*-
"""
@author:https://github.com/labmlai
@description: Parse all files and write to a single file
refer: https://github.com/labmlai/python_autocomplete/blob/master/python_autocomplete/create_dataset.py
"""
import os
import glob
import re
import string
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Set
from typing import Optional
from loguru import logger
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
PRINTABLE = set(string.printable)


def read_file(path: str) -> str:
    """
    Read a file
    """
    with open(path, 'r', encoding='utf8') as f:
        try:
            content = f.read()
        except UnicodeDecodeError as e:
            logger.warning(f"UnicodeDecodeError: {path}, file pass")
            content = ""
    content = ''.join(filter(lambda x: x in PRINTABLE, content))

    return content


def save_file(content, file_path: str):
    with open(file_path, 'w', encoding='utf8') as f:
        f.write(str(content))


def merge_and_save(source_files, path):
    with open(path, 'w', encoding='utf8') as f:
        for src in source_files:
            f.write(read_file(src) + "\n\n")


def get_repos_from_readme(readme_content):
    link_pattern = re.compile(r"""
        \[(?P<title>[^\]]*)\] # title
        \((?P<utl>[^\)]*)\) # url
    """, re.VERBOSE)

    res = link_pattern.findall(readme_content)

    github_repos = []
    repo_pattern = re.compile(r'https://github.com/(?P<user>[^/]*)/(?P<repo>[^/#]*)$')
    for title, url in res:
        repos = repo_pattern.findall(url)
        for r in repos:
            github_repos.append((r[0], r[1]))

    return github_repos


def download_repo(save_dir: str, org: str, repo: str, idx: Optional[int]):
    zip_file = Path(f'{save_dir}/{org}_{repo}.zip')

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


def extract_zip(source_dir, file_path: Path):
    source = Path(source_dir)
    logger.debug(f"Extract {file_path}")
    repo_source = source / file_path.stem
    if repo_source.exists():
        logger.debug(f"Exists: {repo_source}")
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


def get_source_code_by_language(code_languages=("python", "java", "cpp"),
                                save_dir='download/',
                                each_limit_repos=3):
    sources = dict()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(code_languages, str):
        code_languages = [code_languages]
    logger.info(f"Get source code by language: {code_languages}")

    def get_source_files_by_readme(readme_content, sub_save_dir, limit_size):
        zip_dir = sub_save_dir + "/zip"
        src_dir = sub_save_dir + "/src"
        Path(zip_dir).mkdir(parents=True, exist_ok=True)
        Path(src_dir).mkdir(parents=True, exist_ok=True)
        repos = get_repos_from_readme(readme_content)
        if limit_size:
            repos = repos[:limit_size]
        # Download repos
        for i, r in enumerate(repos):
            zip_file = download_repo(zip_dir, r[0], r[1], i)
            if not zip_file:
                continue
            extracted = extract_zip(src_dir, zip_file)
            remove_files(extracted, keep={suffix})
        source_files = glob.glob(f"{src_dir}/**/*{suffix}", recursive=True)
        logger.info(f"Path: {src_dir}/**/*{suffix}, file size: {len(source_files)}")
        return source_files

    if "python" in code_languages:
        logger.debug('Get awesome-python')
        suffix = '.py'
        sub_save_dir = os.path.join(save_dir, 'python')
        readme_file = sub_save_dir + '/README.md'
        if os.path.exists(readme_file):
            readme_content = read_file(readme_file)
        else:
            readme_content = urllib.request.urlopen(
                'https://raw.githubusercontent.com/bharathgs/Awesome-pytorch-list/master/README.md').read()
            readme_content = str(readme_content)
            save_file(readme_content, readme_file)
        sources['python'] = get_source_files_by_readme(readme_content, sub_save_dir, each_limit_repos)
        logger.info(f"Get source code by language: python done")
    if "java" in code_languages:
        logger.debug('Get awesome-java')
        suffix = '.java'
        sub_save_dir = os.path.join(save_dir, 'java')
        readme_file = sub_save_dir + '/README.md'
        if os.path.exists(readme_file):
            readme_content = read_file(readme_file)
        else:
            readme_content = urllib.request.urlopen(
                'https://raw.githubusercontent.com/akullpp/awesome-java/master/README.md').read()
            readme_content = str(readme_content)
            save_file(readme_content, readme_file)
        sources['java'] = get_source_files_by_readme(readme_content, sub_save_dir, each_limit_repos)
        logger.info(f"Get source code by language: java done")
    if "cpp" in code_languages:
        logger.debug('Get awesome-cpp')
        suffix = '.cpp'
        sub_save_dir = os.path.join(save_dir, 'cpp')
        readme_file = sub_save_dir + '/README.md'
        if os.path.exists(readme_file):
            readme_content = read_file(readme_file)
        else:
            readme_content = urllib.request.urlopen(
                'https://raw.githubusercontent.com/fffaraz/awesome-cpp/master/README.md').read()
            readme_content = str(readme_content)
            save_file(readme_content, readme_file)
        sources['cpp'] = get_source_files_by_readme(readme_content, sub_save_dir, each_limit_repos)
        logger.info(f"Get source code by language: cpp done")

    return sources

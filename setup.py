import os
import re
import shutil
from setuptools import setup, find_packages


def get_version(folder_path):
    """Returns the version taken from __init__.py
    Parameters
    ----------
    folder_path : str
        parent folder of __init__.py
    Reference
    ---------
    https://packaging.python.org/guides/single-sourcing-package-version/
    """
    with open(os.path.join(folder_path, '__init__.py'), 'r') as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


def load_readme(folder_path):
    """Returns the text content of the README.md
    Parameters
    ----------
    folder_path : str
        parent folder of README.md
    """
    with open(os.path.join(folder_path, 'README.md'), 'r') as f:
        return f.read()


def clean_repo(repo_folder):
    """Clean the repository before / after installation"""
    dist_folder = os.path.join(repo_folder, 'dist')
    build_folder = os.path.join(repo_folder, 'build')
    if os.path.isdir(dist_folder):
        shutil.rmtree(dist_folder, ignore_errors=True)
    if os.path.isdir(build_folder):
        shutil.rmtree(build_folder, ignore_errors=True)


def main():
    # --------- The following part MUST be completed. --------- #

    # Parent package name, which should (almost) always be 'ibug'
    parent_package_name = 'ibug'
    # Name of your own package, under parent_package_name
    package_name = 'roi_tanh_warping'
    # Repository name on Github
    repository_name = 'roi_tanh_warping'
    # Put the author(s) here
    author = 'Jie Shen'
    # Put the email(s) here
    email = 'js1907@imperial.ac.uk'
    # Short description of your repo
    description = 'Differentiable implementation of some ROI-tanh warping methods.'

    # --------------------------------------------------------- #

    repository_folder = os.path.realpath(os.path.dirname(__file__))
    readme = load_readme(repository_folder)
    version = get_version(os.path.join(repository_folder, parent_package_name, package_name))

    # Please check the final config options carefully
    # If you want to add more options (e.g., package data), please refer to
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/
    config = {
        'name': '_'.join([parent_package_name, package_name]),
        'packages': ['.'.join([parent_package_name, package]) for package in
                     find_packages(os.path.join(repository_folder, parent_package_name))],
        'description': description,
        'long_description': readme,
        'long_description_content_type': 'text/markdown',
        'author': author,
        'author_email': email,
        'version': version,
        'url': 'https://github.com/iBUG-HCI2/' + repository_name,
        'install_requires': ['torch', 'numpy', 'opencv-python'],
        'scripts': [],
        'zip_safe': False,
        'classifiers': [
            'Topic :: Scientific/Engineering',
            'Programming Language :: Python :: 3'
        ],
    }

    clean_repo(repository_folder)
    setup(**config)
    clean_repo(repository_folder)


if __name__ == '__main__':
    main()

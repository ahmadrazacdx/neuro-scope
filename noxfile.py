"""Nox sessions for NeuroScope."""

import shutil
from pathlib import Path

import nox

# Configuration
package = "neuroscope"
python_versions = ["3.12"]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "format",
    "lint",
    "tests",
    "mypy",
    "docs-build",
)


@nox.session(python=python_versions[0])
def format(session):
    """Format code using black and isort."""
    session.install("black", "isort")
    session.run("black", "src", "tests", "docs/conf.py", "noxfile.py")
    session.run("isort", "src", "tests", "docs/conf.py", "noxfile.py")


@nox.session(python=python_versions[0])
def lint(session):
    """Lint code using flake8 and bandit."""
    session.install("flake8", "flake8-bugbear", "flake8-docstrings", "bandit")
    session.run(
        "flake8",
        "src",
        "tests",
        "docs/conf.py",
        "--extend-ignore=D,B950,B905,B007,B001,B028,B904",
    )
    session.run("bandit", "-r", "src", "-ll", "-i")


@nox.session(python=python_versions[0])
def tests(session):
    """Run the test suite."""
    session.install("-e", ".")
    session.install(
        "pytest", "pytest-cov", "coverage", "numpy", "matplotlib", "wcwidth"
    )
    session.run(
        "pytest",
        "--cov=src/neuroscope",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-report=html",
        "-v",
    )


@nox.session(python=python_versions[0])
def mypy(session):
    """Type-check using mypy."""
    session.install("-e", ".")
    session.install("mypy", "numpy", "matplotlib")
    session.run("mypy", "src", "tests")


@nox.session(name="docs-build", python=python_versions[0])
def docs_build(session):
    """Build the documentation."""
    session.install("-e", ".")
    session.install(
        "sphinx",
        "furo",
        "myst-parser",
        "sphinx-copybutton",
        "sphinx-design",
        "sphinxext-opengraph",
        "linkify-it-py",
        "numpy",
        "matplotlib",
        "wcwidth",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", "-b", "html", "-a", "-E", "docs", "docs/_build/html")


@nox.session(python=python_versions[0])
def pre_commit(session):
    """Run pre-commit hooks."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session(python=python_versions[0])
def safety(session):
    """Scan dependencies for insecure packages."""
    session.install("-e", ".")
    session.install("safety")
    session.run("safety", "check")


@nox.session(python=python_versions[0])
def docs(session):
    """Build and serve the documentation with live reloading."""
    session.install("-e", ".")
    session.install(
        "sphinx",
        "sphinx-autobuild",
        "furo",
        "myst-parser",
        "sphinx-copybutton",
        "sphinx-design",
        "sphinxext-opengraph",
        "linkify-it-py",
        "numpy",
        "matplotlib",
        "wcwidth",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", "--open-browser", "docs", "docs/_build/html")

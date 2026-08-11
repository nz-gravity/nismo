# Releasing NISMO

NISMO publishes to PyPI from GitHub Releases through OIDC trusted publishing.
No long-lived PyPI API token belongs in repository secrets.

## One-time setup for the first release

1. Confirm that the PyPI distribution name `nismo` is available or that the
   `nz-gravity` maintainers control it.
2. In the PyPI project or pending-publisher form, configure a GitHub publisher:

   - owner: `nz-gravity`
   - repository: `nismo`
   - workflow: `publish.yml`
   - environment: `pypi`

3. In GitHub repository settings, create an environment named `pypi`. Add
   required reviewers and tag protection if desired. Do not add a PyPI token.
4. Merge the release-preparation pull request only after CI passes.

PyPI supports pending trusted publishers for projects that do not exist yet.
Creating the publisher can therefore create the project on the first successful
upload. Creating the project manually first and then adding the same trusted
publisher is also supported.

## Release checklist

1. Update `src/nismo/_version.py` to the final PEP 440 version.
2. Move user-visible changes from `Unreleased` in `CHANGELOG.md` into a dated
   release section.
3. Update `CITATION.cff` version and release date.
4. Run:

   ```bash
   uv lock --check
   uv run ruff format --check .
   uv run ruff check .
   uv run mypy
   uv run pytest -m "not slow"
   uv build
   uvx twine check dist/*
   ```

5. Verify the wheel in a clean environment and inspect both archives:

   ```bash
   NISMO_RELEASE_CHECK_DIR=$(mktemp -d)
   python -m venv "$NISMO_RELEASE_CHECK_DIR"
   "$NISMO_RELEASE_CHECK_DIR/bin/python" -m pip install dist/nismo-*.whl
   "$NISMO_RELEASE_CHECK_DIR/bin/python" -c "import nismo; print(nismo.__version__)"
   tar -tf dist/nismo-*.tar.gz
   unzip -l dist/nismo-*.whl
   ```

6. Commit and merge the version change.
7. Create a GitHub Release from the exact tag `v<version>`, for example
   `v0.1.0`. The tag must match `nismo.__version__` after removing the leading
   `v`.
8. Publish the GitHub Release. The `Publish to PyPI` workflow validates the tag,
   builds the source and wheel distributions, checks metadata, uploads them as
   a workflow artifact, and publishes from the protected `pypi` environment.
9. Confirm the new version at `https://pypi.org/project/nismo/` and install it
   in a fresh environment.

PyPI files and versions cannot be replaced. If a release artifact is wrong,
increment the version and publish a new release.

# Examples

Runnable Python examples and research notebooks live in this directory.
Start with `phase2_gaussian.py` for the smallest complete MorphZ-backed run.

```bash
python -m pip install "nismo[all]"
python examples/phase2_gaussian.py
```

Before the first PyPI release, install from the repository instead:

```bash
python -m pip install "nismo[all] @ git+https://github.com/nz-gravity/nismo.git@main"
```

## Open notebooks in Colab

- [eggbox.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/eggbox.ipynb)
- [gaussian shell.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/gaussian%20shell.ipynb)
- [gaussian_loggamma_dynesty_nismo.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/gaussian_loggamma_dynesty_nismo.ipynb)
- [peak_sampling.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/peak_sampling.ipynb)

In a fresh Colab runtime, run the repository installation command above, restart
the session, and then run the remaining cells. When testing a branch or fork,
update both the Colab URL and the Git installation ref.

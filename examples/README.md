# Notebook usage

This repository includes Jupyter notebooks under `examples/` and they can be
opened directly in Google Colab.

## Open in Colab

- [eggbox.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/eggbox.ipynb)
- [gaussian shell.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/gaussian%20shell.ipynb)
- [gaussian_loggamma_dynesty_mins.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/gaussian_loggamma_dynesty_mins.ipynb)
- [peak_sampling.ipynb](https://colab.research.google.com/github/nz-gravity/nismo/blob/main/examples/peak_sampling.ipynb)

## Colab setup

Run this in the first cell of a fresh Colab runtime:

```python
%pip install -q "mins[morph,plot,progress] @ git+https://github.com/nz-gravity/nismo.git@main"
```

Then restart the runtime (`Runtime -> Restart session`) and run all cells.

If you are using a fork or non-main branch, update both:

- the Colab URL path (`github/<owner>/<repo>/blob/<branch>/...`)
- the install target (`git+https://github.com/<owner>/<repo>.git@<branch>`)

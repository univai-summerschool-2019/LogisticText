# LogisticText
The best way to work with this data is to use the class [jupyter-hub](https://jhub2.univ.ai/) instance.

## Local Installation
This is subdivided into two categories, but does not cover how you will obtain
`python` (`virtualenvs` are recommended!).

### Anaconda

This may not work very will in non-POSIX shells:

``` bash
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

### Pip

By far the easiest way to work with packages without specialized tools:

``` bash
pip install -r requirements.txt
```

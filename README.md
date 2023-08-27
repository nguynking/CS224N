<h1 align="center">CS224N: Natural Language Processing with Deep Learning</h1>
<p align="center"><i>Stanford - Winter 2023</i></p>

## About

### Overview

These are my solutions for the **CS224N** course assignments offered by _Stanford University_ (Winter 2023). Written questions are explained in detail, the code is brief and commented. These solutions are heavily inspired by [mantasu's repo](https://github.com/mantasu/cs224n/tree/main) and [floriankark's repo](https://github.com/floriankark/cs224n-win2223/tree/main).

### Main sources (official)
* [**Course page**](http://web.stanford.edu/class/cs224n/index.html)
* [**Assignments**](http://web.stanford.edu/class/cs224n/index.html#schedule)
* [**Lecture videos** (2021)](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)

## Requirements
For **conda** users, the instructions on how to set-up the environment are given in the handouts. For `pip` users, I've gathered all the requirements in one [file](requirements.txt). Please set up the virtual environment and install the dependencies (for _linux_ users):

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

You can install everything with **conda** too (see [this](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib)). For code that requires **Azure** _Virtual Machines_, I was able to run everything successfully on **Google Colab** with a free account.

> Note: Python 3.8 or newer should be used

## Solutions

### Structure

For every assignment, i.e., for directories `assigment1` through `assignment5`, there is coding and written parts. The `solutions.pdf` files are generated from latex directories where the provided templates were filled while completing the questions in `handout.pdf` files and the code.

### Assignments

* [**A1**](assignment1): Exploring Word Vectors (_Done_)
* [**A2**](assignment2): word2vec (_Done_)
* [**A3**](assignment3): Dependency Parsing (_Done_)
* [**A4**](assignment4): Neural Machine Translation with RNNs and Analyzing NMT Systems (_Done_)
* [**A5**](assignment5): Self-Attention, Transformers, and Pretraining (_Done_)

## Future works
- Complete the minBERT project.

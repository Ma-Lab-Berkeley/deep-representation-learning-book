# "Learning Deep Representations of Data Distributions"


This repository is the _source code_ for the book "Learning Deep Representations of Data Distributions". 
If you just want to read the book, _you should not need this repository_. We have a copy at `INSERT LINK` that will be updated periodically.

Generally, you should be accessing the source code for one of the following purposes:
- You want to build the book, or one of its chapters, from scratch. See [this section](#building-the-book-or-chapter).
- You want to use the source code that generated a figure in the book. See [this section](#using-the-code).
- You want to ask a question, or tell us that something in the repository doesn't work quite right. See [this section]().
- You want to contribute some content, for example a translation, or some technical content within one or more chapters. See [this section](#making-a-contribution).

## Building the Book or a Chapter

### Prerequisites for Building the Book or a Chapter

- A local `pdflatex` distribution ([TexLive](https://www.tug.org/texlive/) is a recommended choice)
- A binary of `latexmk`, which comes with `TexLive` but can alternatively be installed using `brew`    

### How to Build the Book or a Chapter

To build the book, navigate to the repository base folder and run:
```
latexmk book-main.tex
```
and read `book-main.pdf`.

You can also build individual chapters via `latexmk`! To build a chapter, say Chapter 2, you can navigate to `chapters/chapter2` and run:
```
latexmk classic-models.tex
```
and read `classic-models.pdf`. (`classic-models.tex` is the chapter file for Chapter 2). We did some LaTeX hacks so that compiling the main book or individual chapters should work in your IDE (please file an issue if it doesn't work).

## Using the Code

### Prerequisites for Using the Code

- The Python package manager [`uv`](https://docs.astral.sh/uv/)

### How to Use the Code

- Navigate to the repository and run `uv sync`, which creates a Python virtual environment restricted to the repository.
- Go to the chapter folder that you are interested in and navigate to the code directory, i.e., `chapter3/code`.
- Run the desired file with `uv run <filename>` and observe the outputs.

## Raising an Issue

### Prerequisites for Raising an Issue

A GitHub account.

### How to Raise an Issue

Use the GitHub "Raise Issue" UI, [linked here](https://github.com/Ma-Lab-Berkeley/ldrdd-book/issues). The set of all issues is found [here](https://github.com/Ma-Lab-Berkeley/ldrdd-book/issues). Please give us as much detail as possible when making your issue.
- If you are raising an issue about any of the code, please let us know your system details: OS, Python version, `uv` version, the configuration of any GPUs/TPUs you are using, etc.
- If you are raising an issue about the content, please be as descriptive as possible. If you believe that something is wrong, please give a concrete reason why (e.g. a counterexample).

**Please use English** as this maximizes the number of people who can help.

## Making a Contribution

### Prerequisites for Making a Contribution

- Minimally, a text editor and GitHub account. 
- But you probably also want to be able to [build the book and/or chapter](#building-the-book-or-a-chapter) and/or [edit and run the code](#using-the-code).

### How to Make a Contribution

- Make a new branch on GitHub, and call it whatever you like (among names that are not taken), e.g., `my_new_branch`.
- Clone the repository locally using `git clone`:
```
git clone <PLACEHOLDER URL>
```
- Change the branch, i.e.,
```
git checkout -b my_new_branch
```
- Make whatever edits you want, then add them, commit, and push to GitHub:
```
git add <some files you changed>
git commit -m "<descriptive commit message>"
git push -u origin my_new_branch
```
- You can commit and push as many times as you want until you are satisfied.
- Make a pull request to `main` using the [GitHub UI](https://github.com/Ma-Lab-Berkeley/ldrdd-book/compare). The pull request will likely go through some revision from a core maintainer, and may be merged to `main` afterwards.

**Please use English** as this maximizes the number of people who can attend to your contribution.

### General Guidelines for Contributing

- For code: please examine current code files and see the coding patterns.
- For LaTeX: please examine current LaTeX files and see the coding patterns.

Note: While this repository is private (i.e., "beta version") you are technically allowed to push to `main` directly. **Please do not.**

## Citation Information

Many thanks!

```
TODO: PLACEHOLDER
```

## INTERNAL USE

### TODOs
- finish the rest of the readme
- fix code

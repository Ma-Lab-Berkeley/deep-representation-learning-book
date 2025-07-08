# "Learning Deep Representations of Data Distributions"


This repository is the _source code_ for the book "Learning Deep Representations of Data Distributions". 
If you just want to read the book, _you should not need this repository_. We have a copy at `INSERT LINK` that will be updated periodically.

Generally, you should be accessing the source code for one of the following purposes:
- You want to build the book, or one of its chapters, from scratch. See [this section](#building-the-book-or-chapter).
- You want to contribute some content, for example a translation, or some technical content within one or more chapters. See 
- You want to use the source code that generated a figure in the book.

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

To build a chapter, say Chapter 2, _stay in the repository base folder_ and run:
```
latexmk chapters/chapter2/classic-models.tex
```
and read `chapters/chapter2/classic-models.pdf`.

## Using the Code

### Prerequisites for Using the Code


### How to Use the Code


## Making a Contribution

### Prerequisites for Making a Contribution

- Minimally, a text editor and GitHub account. 
- But you probably also want to be able to [build the book and/or chapter](#) or [edit and run the code]().

### How to Make a Contribution

### General Guidelines for Contributing


## Citation Information

Many thanks!

```
TODO: PLACEHOLDER
```

## INTERNAL USE

### TODOs
- Fix relative imports to get subfiles to compile without errors. [helpful link 1](https://tex.stackexchange.com/questions/36175/what-do-newcommand-renewcommand-and-providecommand-do-and-how-do-they-differ) +  [helpful link 2](https://tex.stackexchange.com/questions/107064/bibliographies-when-using-subfiles)
- finish the rest of the readme
- fix code
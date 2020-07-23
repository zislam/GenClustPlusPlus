# GenClustPlusPlus

Class implementing GenClust++ clustering algorithm. For more information, see:

*Islam, M. Z., Estivill-Castro, V., Rahman, M. A. and Bossomaier, T. (2018). Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering. Expert Systems with Applications.*

## BibTeX
```
\article{adnan2017forest,
  title={Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering},
  author={Islam, M. Z., Estivill-Castro, V., Rahman, M. A. and Bossomaier, T.},
  journal={Expert Systems with Applications},
  year={2018},
  volume={91},
  pages={402-417},
  publisher={Elsevier}
}
```

## Installation

Either download GenClustPlusPlus from the Weka package manager, or download the latest release from the "Releases" section on the sidebar of Github. A video on the installation and use of the package can be found [here](https://www.youtube.com/watch?v=WfETv17gdbY&t=0s).

## Compilation / Development

Set up a project in your IDE of choice, including weka.jar as s compile-time library.

## Valid options are:

`-G <num generations>`
 Number of generations for genetic algorithm.
 (default 60)

`-P <popn size>`
 Initial population size for generic algorithm.
 (default 30)

`-N <max num iterations>`
 Max iterations for initial k-means.
 (default 60)

`-Q <max num iterations>`
 Max iterations for quick k-means.
 (default 15)

`-F <max num iterations>`
 Max iterations for final run of k-means.
 (default 50)

`-D <duplicate threshold>`
 Threshold for difference between two genes for them to be considered
 duplicates. Always between 0 and 1.
 (default 0)

`-M`
 Do not replace missing values with a global mean / mode.
 (default false)

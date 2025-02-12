# MatrixAlgebraKit.jl

**A Julia interface for matrix algebra, with a focus on performance, flexibility and extensibility.**

## Table of contents

```@contents
Pages = ["interface.md", "library.md"]
Depth = 1
```

## Installation

MatrixAlgebraKit.jl is part of the general registry, and can be installed through the package manager as:

```
pkg> add MatrixAlgebraKit
```

## Key features

The main goals of this package are:
* Definition of a common interface that is sufficiently expressive to allow easy adoption and extension.
* Ability to pass pre-allocated output arrays where the result of a computation is stored.
* Ability to easily switch between different backends and algorithms for the same operation.
* First class availability of pullback rules that can be used in combination with different AD ecosystems.

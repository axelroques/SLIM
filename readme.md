# Side-Length-Independent Motif (SLIM) algorithm

Python implementation of a multivariate time series motif discovery and 'volatility' analysis algorithm (Side-Length-Independent Motif).

Inspired by _Cartwright et. al. (2022)_.

This algorithm has three main steps:

- Symbolic Aggregate approXimation (SAX). _Introduced in Lin et. al. (2003)_, SAX is a symbolic representation of time series. Each time series of the input dataframe is transformed into a symbolic representation using the SAX algorithm (https://github.com/axelroques/SAX). Rather than discretizing the whole time series, subsequences of length `n` are extracted from the time series, normalized and converted into a SAX word, with a stride of `k`. Each subsequence is thus discretized individually using SAX, and all of these SAX words are concatenated to form one single SAX phrase. This step differs from _Cartwright et. al._: they perform SAX on whole time series rather than overlapping subsequences.
- From this SAX phrase, numerosity reduction is employed: if a word occurs multiple times consecutively, we only keep its first occurrence. _E.g._, the phrase '123 123 123 122 122 132 122' becomes '123 122 132 122'. Numerosity reduction is the key that makes variable-length motif discovery possible. Note that this step also differs from _Cartwright et. al._. _Cartwright et. al._ perform numerosity reduction - or rather they apply the MDL principle as they call it - on individual symbols. Here, the numerosity reduction is done on 'patterns' of symbols, similarly to _Li et. al. (2013)_. The effect should be substantially identical - I believe - but discretizing overlapping subsequences should increase the accuracy of the symbolic representation.
- Matrix Profile (MP). Implementation of the MP is done using the mSTOMP Python implementation from the authors of _Yeh et. al. (2017)_(https://github.com/mcyeh/mstamp/tree/master/Python). More information on the Matrix Profile can be found here: https://www.cs.ucr.edu/~eamonn/MatrixProfile.html. The input parameter `window_length` corresponds to the window length used when computing the MP and should roughly equal the length of the expected motifs. Note that because the MP uses the Euclidian distance at its core, we need to use numbers as symbols for the discretization.

_Remark: Using 'patterns' rather than unitary symbols for the numerosity reduction step could potentially create bias and favor some patterns, because the Matrix Profile uses the Euclidian distance in its computations. I do not know the extent of this issue!_

## Requirements

**Mandatory**:

- numpy
- SAX (https://github.com/axelroques/SAX)

**Optional**:

- matplotlib

---

## Examples

**Processing the data**

```python
df = pd.read_csv('your_data.csv')
slim = SLIM(df, w=3, a=4, n=100, k=10, window_length=4)
slim.process()
```

The `slim.sax` object contains results from the different steps of the SAX algorithm and the various SAX parameters:

- `slim.sax.df_INT` returns the dataframe after the normalization step.
- `slim.sax.df_PAA` returns the dataframe after the PAA step.
- `slim.sax.df_SAX` returns the dataframe after the discretization step.
- `slim.sax.w` returns the number of segments in the PAA - and SAX - representation (after the dimensionality reduction).
- `slim.sax.a` returns the number of symbols in the alphabet.
- `slim.sax.alphabet` returns the different symbols in the alphabet (determined by parameter _a_).
- `slim.sax.breakpoints` returns the values of the different breakpoints computed to discretize the time series.

`slim.SAX_phrases` contains the SAX phrase generated from the discretization of the subsequences.

`slim.reduced` contains the 'reduced' representation of the SAX phrase after the numerosity reduction step.

**Find the positions of k-dimensional motifs**

```python
slim.get_motifs(dim=3)
```

`dim` corresponds to the number of dimensions that the motifs pan.

**Plot the motifs**

```python
slim.show_motifs(level=1)
```

`level` is an arbitrary term used to describe which bins the motif belong to. To be more precise, the Matrix Profile is supposed to be used to find joins in time series, _i.e._ pairs of similar motifs. Here, the Matrix Profile values are binned according to their value: lower values populate the lower bins and _vice versa_. I chose - arbitrarily - to return the positions of the 5 lowest bins in function `get_motifs`. `level` simply corresponds to the bin number and ranges from 0 to 4.


from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from sax import SAX_subsequences
from.mstamp import mstamp


class SLIM:

    def __init__(self, df,  w=3, a=4, n=100, k=10, window_length=50):

        # Raw data
        self.df = df

        # SAX parameters
        self.sax = SAX_subsequences(df, w=w, a=a, n=n, k=k,
                                    alphabet_type='numbers')

        # MP parameters
        self.window_length = window_length

        # Some parameters that will be used when plotting
        self.sax_time_vectors = None

    def process(self):
        """
        Run the SLIM algorithm process. 
        """

        # SAX subsequence discretization
        self.sax.process()

        # Generate SAX phrases
        self.SAX_phrases = self.generate_SAX_phrases(self.sax.df_SAX)

        # Numerosity reduction
        self.reduced, self.mapping = self.numerosity_reduction(
            self.SAX_phrases)

        # Create an array from the reduced representations
        reduced_array = self.create_array(self.reduced)

        # Compute the Matrix Profile
        self.MP, self.MP_indices, self.dims = self.compute_MP(
            reduced_array, self.window_length)

        return

    @staticmethod
    def generate_SAX_phrases(df_SAX):
        """
        Generate SAX phrases from the SAX words for 
        each column in the input dataframe.
        """

        SAX_phrases = {
            f'{col}': [''.join(str(c) for c in subsequence.loc[:, col])
                       for subsequence in df_SAX]
            for col in df_SAX[0].columns[1:]
        }

        return SAX_phrases

    @staticmethod
    def numerosity_reduction(SAX_phrases):
        """
        Conducts the numerosity reduction step, i.e.,
        if a string occurs multiple times consecutively,
        records only its first occurrence.

        Also creates a hash table called mapping, that will 
        become useful to map the position of the SAX words
        from their 'reduced' representation back to their
        original representation. 
        """

        mapping = {
            f'{col}': [] for col in SAX_phrases.keys()
        }
        reduced = {
            f'{col}': [] for col in SAX_phrases.keys()
        }

        # Iterate over the columns
        for key in SAX_phrases.keys():

            # Iterate over the SAX words
            for i, word in enumerate(SAX_phrases[key]):

                # If we are at the first iteration
                if i == 0:

                    # Add word to the reduced list
                    reduced[key].append(int(word))

                    # Add starting position of the word
                    mapping[key].append(i)

                # Add word to the reduced list if it was
                # not added right before
                if int(word) != reduced[key][-1]:
                    reduced[key].append(int(word))

                    # Add starting position of the word
                    mapping[key].append(i)

        return reduced, mapping

    @staticmethod
    def create_array(reduced):
        """
        Creates a numpy array from the reduced representation
        of the different time series.

        Because the time series may have variable length, we 
        set the length of the numpy array to the max length 
        of the reduced representations and pad the shorter time
        series with zeros.
        """

        # Max length of the reduced representation
        reduced_list = [reduced[key] for key in reduced.keys()]
        max_length = max(map(len, reduced_list))

        reduced_array = np.array(
            [r + [0]*(max_length-len(r)) for r in reduced_list],
            dtype=float)

        # print('reduced_array =', reduced_array)
        return reduced_array

    @staticmethod
    def compute_MP(reduced_array, window_length):
        """
        Computes the Matrix Profile of a multidimensional
        time series.

        The function only accepts a numpy array as an input. Each row 
        represents data from a different dimension while each 
        column represents data from the same dimension.

        The window_size parameter represents the estimation of
        how many data points might be found in a pattern.
        """

        # MP, MP_indices = stumpy.mstump(reduced_array, m=window_length)

        MP, MP_indices, dims = mstamp(reduced_array, window_length,
                                      return_dimension=True)

        # Remove values where indices =-1
        MP_pruned = []
        MP_indices_pruned = []
        for dim in range(len(MP)):

            index = np.where(MP_indices[dim] == -1)[0]

            if index.any():
                MP_pruned.append(MP[dim][:index[0]])
                MP_indices_pruned.append(MP_indices[dim][:index[0]])
            else:
                MP_pruned.append(MP[dim])
                MP_indices_pruned.append(MP_indices[dim])

        return MP_pruned, MP_indices_pruned, dims

    def get_motifs(self, dim=2):
        """
        Retrieve motifs.
        """

        # Retrieve correct variables for the search
        MP = self.MP[dim-1]
        MP_indices = self.MP_indices[dim-1]
        dims = self.dims[dim-1]

        # Find the position of the motifs
        motifs = self.find_motifs_position(MP, MP_indices, dims=dims)

        # Reverse the numerosity reduction
        motifs_indices = self.find_true_motifs_positions(self.mapping, motifs,
                                                         self.window_length)

        # Find the time values associated with these indices
        if not self.sax_time_vectors:
            self.sax_time_vectors = [df['t'].tolist()
                                     for df in self.sax.df_SAX]

        self.time_indices = self.find_time_values(
            self.sax_time_vectors, motifs_indices)

        return

    def show_motifs(self, level):
        """
        Display motifs.
        """

        if self.time_indices:
            self.plot_motifs(self.sax.df, self.time_indices, level)
        else:
            raise RuntimeError('Run get_motifs method first.')

        return

    @staticmethod
    def find_motifs_position(MP, MP_indices, dims):
        '''
        Finds the motif associated with the lowest matrix profile value
        with a dimensionality of dimension.
        '''

        # Assign each value of MP to a bin
        bins = np.linspace(MP.min(), MP.max(), num=100)
        digitized = np.digitize(MP, bins)

        # Retrieve the first 5 bins
        selected_bins = np.unique(digitized)[:5]
        motifs = {
            'indices': [],
            'dimensions': []
        }

        for bin in selected_bins:

            # Get indices from digitized array
            ind = np.where(digitized == bin)

            # Transform these indices into real indices
            # using MP_indices
            motifs['indices'].append(MP_indices[ind])

            # Also store the relevant dimesions involved
            # with these indices
            motifs['dimensions'].append([tuple(dims[:, i]) for i in ind[0]])

        return motifs

    @staticmethod
    def find_true_motifs_positions(mapping, motifs, window_length):
        """
        Find the position of the motif in the original real-valued
        time series.

        We must take into account the numerosity reduction using the
        mapping dictionary. 

        The returned motifs_indices architecture is the following:
        level i -> 'col_name' -> list of real positions like 
        (i_start, i_end).

        Uses a defaultdict to dynamically add the positions to the
        dictionary.
        """

        # Create dictionary linking col number and col name
        int2col = {i: col for i, col in enumerate(mapping.keys())}

        motifs_indices = {
            f'level {l}': defaultdict(list) for l in range(len(motifs['indices']))
        }

        # Iterate over the motifs info
        for l, (positions, dims) in enumerate(zip(motifs['indices'],
                                                  motifs['dimensions'])):

            for pos, dim in zip(positions, dims):

                for d in dim:

                    motifs_indices[f'level {l}'][int2col[d]].append(
                        tuple((mapping[int2col[d]][pos],
                               mapping[int2col[d]][pos+window_length]))
                    )

        return motifs_indices

    @staticmethod
    def find_time_values(sax_time_vectors, motifs_indices):
        """
        Returns the time values associated with the 
        motifs indices.
        """

        # Same shape as motifs_indices
        time_indices = {
            key: {subkey: [] for subkey in motifs_indices[key]}
            for key in motifs_indices.keys()
        }

        # Iterate over the levels
        for key in motifs_indices.keys():

            for subkey in motifs_indices[key]:

                for indices in motifs_indices[key][subkey]:

                    t_start = sax_time_vectors[indices[0]][0]
                    t_end = sax_time_vectors[indices[1]][-1]

                    # Add some time to the end because the SAX
                    # algorithm only computes the starting time
                    # of a segment
                    t_end += sax_time_vectors[indices[1]][1] - \
                        sax_time_vectors[indices[1]][0]

                    time_indices[key][subkey].append(tuple((t_start, t_end)))

        return time_indices

    @staticmethod
    def plot_motifs(df, time_indices, level):
        """
        Plots the motifs using the time indices.
        """

        n_row = len(df.columns[1:])
        _, axes = plt.subplots(n_row, 1,
                               figsize=(n_row*3, 8))

        # Simple plot of the whole time series
        for ax, col in zip(axes.ravel(), df.columns[1:]):
            ax.plot(df['t'], df.loc[:, col], c='k', alpha=0.5)
            ax.set_title(col)

        # Restrain plot to the specified level
        time_indices = time_indices[f'level {level}']

        # Motifs plots
        for col in time_indices.keys():

            # Ax number
            i_ax = df.columns.get_loc(col) - 1

            # Get sub-dataframe using the time indices
            for indices in time_indices[col]:

                cut = df.loc[(df['t'] >= indices[0]) &
                             (df['t'] < indices[1])]

                axes[i_ax].plot(cut['t'], cut.loc[:, col],
                                c='royalblue', alpha=0.8)

        plt.tight_layout()
        plt.show()

        return

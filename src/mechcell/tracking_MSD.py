import numpy as np


def displacement(df, cid):
    df_cid = df.loc[df["Condensate ID"] == cid]

    xs = df_cid["x_corr"].values
    ys = df_cid["y_corr"].values
    frames = df_cid["t"].values

    n_frames = len(frames)
    ds = np.zeros(n_frames)

    for i in range(n_frames - 1):
        ds[i + 1] = np.abs(xs[i] - xs[i + 1]) + np.abs(ys[i] - ys[i + 1])

    return frames, ds


class MSDAnalyzer:
    def __init__(self, tracks):
        """
        Initialize the MSDAnalyzer with tracking data.
        :param tracks: A pandas DataFrame with tracking data containing
                       'Condition', 'Condensate ID', 't', 'x_corr', 'y_corr' columns.
        """
        self.tracks = tracks

    def calculate_interval_lengths(self, cond_frames):
        """
        Calculate the interval lengths (with overlapping intervals) of visible condensates.

        Input: Frames where the condensate is visible
        Output: A dictionary of number of possible interval lengths where the condensates are visible
        """
        # Max number of frames
        max_intervals = np.max(cond_frames)

        # Initiate dictionary
        interval_lengths = {}

        for k in range(1, max_intervals):
            interval = [(i, j) for i, j in zip(cond_frames, cond_frames[k:])]
            length = len(interval)
            if length > 0:
                interval_lengths[k] = length

        return interval_lengths

    def calculate_MSD(self, interval_lengths, x_corr, y_corr):
        """
        Calculate the Mean Squared Displacement (MSD) over specified intervals.

        This method computes the MSD, which is a measure of the average squared
        distance that particles in a system have moved over time. The calculation
        is performed over specified time intervals for given x and y position
        coordinates.

        Parameters:
        ----------
        interval_lengths : dict
            A dictionary where keys are interval lengths that specify the time
            intervals over which MSD is computed.
        x_corr : list or numpy.ndarray
            A list or array of x-coordinates of the particle positions over time.
        y_corr : list or numpy.ndarray
            A list or array of y-coordinates of the particle positions over time.

        Returns:
        -------
        all_MSDs : list
            A list of MSD values computed for each specified interval length.
        all_MSDs_dict : dict
            A dictionary mapping each interval length key to its computed MSD value.
        MSD_intervals_all : list of lists
            A list containing lists of all MSD intervals for each specified interval
            length. Each sublist contains MSD values computed for each possible
            segment within a particular interval length.

        Notes:
        -----
        - This function assumes `x_corr` and `y_corr` are of the same length.
        """
        # Initiate lists
        all_MSDs = []
        MSD_intervals_all = []

        for key in interval_lengths:
            MSD_intervals = []

            # Compute MSD values for the intervals
            for i in range(key, len(x_corr)):
                # Check if the index is non-negative
                if i - key >= 0:
                    MSD_interval = (x_corr[i] - x_corr[i - key]) ** 2 + (y_corr[i] - y_corr[i - key]) ** 2
                    MSD_intervals.append(MSD_interval)

            MSD_intervals_all.append(MSD_intervals)

            MSD_sum = np.sum(MSD_intervals)
            msd = MSD_sum / len(MSD_intervals)
            all_MSDs.append(msd)

        return all_MSDs

    def process_condensates(self, tracks):
        """
        Process the Mean Squared Displacement (MSD) values for each condensate.

        This method takes a DataFrame containing tracking information for multiple
        condensates, processes each individual condensate by calculating the MSD
        for its motion data over multiple time intervals, and returns a list of
        MSD results for all condensates.

        Parameters:
        ----------
        tracks : pandas.DataFrame
            A DataFrame with tracking data for condensates, which must contain
            the following columns:
            - "Condensate ID": Unique identifier for each condensate.
            - "x_corr": Corrected x-coordinates of the condensate over time.
            - "y_corr": Corrected y-coordinates of the condensate over time.
            - "t": Time points corresponding to the condensate positions.

        Returns:
        -------
        msd_cond : list
            A list where each element is the MSD computation for a given condensate,
            as derived from the MSD calculation for multiple intervals.

        Notes:
        -----
        - This function assumes that each condensate is uniquely identified within
        the `tracks` DataFrame by the "Condensate ID" column.
        - The calculation of MSD is performed for each unique condensate using the
        coordinates and time data retrieved from the DataFrame.
        """
        # Initiate list for msd values
        msd_cond = []

        # Extract condensate ids
        condensate_ids = np.unique(tracks["Condensate ID"])

        for condensate_id in condensate_ids:
            # Select data for the current condensate
            selected_cond = tracks.loc[tracks["Condensate ID"] == condensate_id]

            # Retrieve coordinates and time for current condensate
            x_corr = selected_cond["x_corr"].to_numpy()
            y_corr = selected_cond["y_corr"].to_numpy()
            time = selected_cond["t"].tolist()

            # Calculate interval lengths
            interval_lengths = self.calculate_interval_lengths(time)

            # Calculate MSD for all intervals for the current condensate
            msd = self.calculate_MSD(interval_lengths, x_corr, y_corr)

            msd_cond.append(msd)

        return msd_cond

    def extract(self, msd_cond):
        """
        Extract and organize Mean Squared Displacement (MSD) values by position index.

        This method processes a list of MSD sublists, each representing the
        MSD values calculated over different intervals for condensates.
        It reorganizes these values into a list of lists (`cond_res`), where each
        sublist at a given index contains the MSD values from all condensates for
        that specific interval index. This allows for easy comparison of MSD values
        across condensates for the same interval index.

        Parameters:
        ----------
        msd_cond : list of lists
            A list where each element is a sublist representing MSD values
            for various given intervals (by position) for a single condensate.

        Returns:
        -------
        cond_res : list of lists
            A list of lists where each inner list contains the MSD values for a
            specific position index across all condensates, enabling comparison
            and analysis of MSD trends across condensates.

        Notes:
        -----
        - The function assumes that `msd_cond` is a non-empty list where each
        sublist corresponds to MSD values of a separate condensate with possibly
        different lengths.
        - The output `cond_res` is constructed such that each sublist corresponds
        to the MSD values of all condensates at the same relative position index
        of their respective lists.
        """

        # Find maximum sublist length in msd_cond
        max_l = max(len(sub) for sub in msd_cond)

        # Initialize an empty list for each position
        cond_res = [[] for _ in range(max_l)]

        # Fill up the cond_res list with corresponding MSD values
        for sub in msd_cond:
            for i, item in enumerate(sub):
                cond_res[i].append(item)

        return cond_res

    def mean_MSD(self, cond_res):
        """
        Compute the mean Mean Squared Displacement (MSD) for each position index across condensates.

        Parameters:
        ----------
        cond_res : list of lists
            A list where each sublist contains MSD values from different
            condensates at the same interval index position.

        Returns:
        -------
        mean_cond : list
            A list containing the mean MSD value for each interval index
            across all condensates. Each element in the list corresponds
            to the average MSD at a given interval index.
        """

        mean_cond = []

        for msd in cond_res:
            mean_cond.append(np.mean(msd))

        return mean_cond

    def error_MSD(self, cond_res):
        """
        Compute the standard deviation of Mean Squared Displacement (MSD) values for each position index across condensates.

        Parameters:
        ----------
        cond_res : list of lists
            A list where each sublist contains MSD values from different
            condensates at the same interval index position.

        Returns:
        -------
        err_cond : list
            A list containing the standard deviation (error) of MSD values for
            each interval index across all condensates. Each element in the list
            represents the variability of MSD values at a given interval index.
        """

        err_cond = []

        for err in cond_res:
            err_cond.append(np.std(err))

        return err_cond

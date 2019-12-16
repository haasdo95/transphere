import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import time

from ghcn_helper import *
from process_laplacian import *


class GHCN(Dataset):
    def __init__(self, date_start, date_end, elems_wanted, graph_freezing):
        """
        load stuff up here for performance
        :param date_start: like "20180101"
        :param date_end: like "20181231"
        :param graph_freezing: determines how often the graph has to be rebuilt.
                               if graph_freezing=7, graph will be built by union of stations in future 7 days
        """
        # config stuff
        self.date_start = date_start
        self.date_end = date_end
        self.graph_freezing = graph_freezing
        self.elems_wanted = elems_wanted
        # load stuff up; this can take a little while
        self.yearbook = build_yearbook(date_start, date_end, elems_wanted)
        self.df_by_date = list(iterate_stations(date_start, date_end, self.yearbook))
        # things to keep track of during training
        self.current_laplacian = None  # the M^-1 @ L product
        self.omega_stations = None  # stations used to build graph; the Omega set in the frozen time period
        self.blacklist = None
        self.days2expire = 0  # will be reset to graph_freezing when getting zero

    def __getitem__(self, idx):
        date_str, df = self.df_by_date[idx]
        # check graph expiration
        if self.days2expire == 0:  # brand new freezing period; everything is None
            print("---previous Laplacian expired or no previous Laplacian; building new---")
            # Very easy to make off-by-one error here; Remember that date ranges are inclusive
            right_bound = days_after(date_str, self.graph_freezing-1)
            if str2date(right_bound) > str2date(self.date_end):  # if right_bound comes later than date_end
                print("---warning: exceeding bound; not building graph with union of all {} days---"
                      .format(self.graph_freezing))
                right_bound = self.date_end  # resolve boundary case
            self.omega_stations = find_all_stations(date_str, right_bound, self.yearbook)
            # NOTE: those blacklisted shouldn't be in ANYTHING
            _, self.blacklist = pre_coarsen(self.omega_stations, 0.01, inplace=True)
            L, M = compute_cotan_laplacian(self.omega_stations)
            laplacian = combine_cotan_mass(L, M)
            lmax = compute_lmax(laplacian)
            self.current_laplacian = scale_laplacian(laplacian, lmax)  # scale it right here...
            assert self.current_laplacian.format == "coo"
            self.days2expire = self.graph_freezing
        self.days2expire -= 1
        # remove from current df whatever is in blacklist; not inplace
        # errors="ignore" makes sure that no errors raised even if some stations in blacklist are not in df
        df = df.drop(self.blacklist, axis=0, errors="ignore")
        # df = df.drop(self.blacklist, axis=0)

        assert set(df.index).issubset(set(self.omega_stations.index))
        # form the input feature x, shaped as (V, Fin)
        # if doing dense regression, perhaps a good idea to include label in the last column of x as well
        same_indices = []
        diff_indices = []
        x = torch.zeros(size=(len(self.omega_stations), len(self.elems_wanted)), dtype=torch.double)
        for i in range(len(self.omega_stations)):
            if self.omega_stations.index[i] in df.index:  # has record in df
                # x[i, :] = torch.from_numpy(df.loc[self.omega_stations.index[i]][self.elems_wanted].to_numpy())
                same_indices.append(i)
            else:  # no record in df; fill with zero
                diff_indices.append(i)
        # THIS indexing trick saves a great amount of time
        # check integrity
        for i, si in enumerate(same_indices):
            assert self.omega_stations.index[si] == df.index[i]

        x[same_indices, :] = torch.from_numpy(df[self.elems_wanted].values)
        # mask the entries in self.current_laplacian; mask whatever is in curr_laplacian but not in df
        masked_laplacian = mask_laplacian(self.current_laplacian, np.array(diff_indices, dtype=np.long))
        assert masked_laplacian is not self.current_laplacian  # make sure deep copy
        assert masked_laplacian.format == "coo"

        masked_laplacian = scipy2torch(masked_laplacian)
        # DESIGN DECISION: for impainting, some more masking to do externally

        return masked_laplacian, x

    def __len__(self):
        date_range = pd.date_range(self.date_start, self.date_end)
        return len(date_range)

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import random
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
        non_zero_indices = []
        zero_indices = []
        x = torch.zeros(size=(len(self.omega_stations), len(self.elems_wanted)), dtype=torch.double)
        for i in range(len(self.omega_stations)):
            if self.omega_stations.index[i] in df.index:  # has record in df
                # x[i, :] = torch.from_numpy(df.loc[self.omega_stations.index[i]][self.elems_wanted].to_numpy())
                non_zero_indices.append(i)
            else:  # no record in df; fill with zero
                zero_indices.append(i)
        # check integrity
        for i, si in enumerate(non_zero_indices):
            assert self.omega_stations.index[si] == df.index[i]
        non_zero_indices = np.asarray(non_zero_indices, dtype=np.long)
        zero_indices = np.asarray(zero_indices, dtype=np.long)
        # fill-in non-zero entries in x; THIS indexing trick saves a great amount of time
        x[non_zero_indices, :] = torch.from_numpy(df[self.elems_wanted].values)
        # mask the entries in self.current_laplacian; mask whatever is in curr_laplacian but not in df
        masked_laplacian = mask_laplacian(self.current_laplacian, zero_indices)
        assert masked_laplacian is not self.current_laplacian  # make sure deep copy
        assert masked_laplacian.format == "coo"
        # scipy2torch can wait...
        # masked_laplacian = scipy2torch(masked_laplacian)
        # DESIGN DECISION: for impainting, some more masking to do externally
        # for that, also returns the indices in self.omega_stations that have value
        return masked_laplacian, x, non_zero_indices

    def __len__(self):
        date_range = pd.date_range(self.date_start, self.date_end)
        return len(date_range)


class Sampler:
    def __init__(self, times, size):
        self.times = times
        self.size = size

    def sample(self, indices):
        """
        samples for $times$ times with each sample size = $size$
        samples the indices that should be masked
        :param indices:
        :return:
        """
        if type(self.size) is int and self.size >= 1:
            num_sampled = self.size
        elif type(self.size) is float and self.size < 1:
            num_sampled = int(len(indices) * self.size)
        else:
            raise Exception("self.size not legit!")
        idx_to_mask = random.sample(indices, num_sampled)
        assert len(idx_to_mask) == num_sampled
        return idx_to_mask


class GHCNImpainter(Dataset):
    def __init__(self, ghcn, sampler):
        """
        :param ghcn: a ghcn dataset object
        :param sampler: a sampler
        """
        self.ghcn = ghcn
        self. sampler = sampler
        # things to keep track of
        self.masked_laplacian, self.x, self.non_zeros_indices = None, None, None

    def __len__(self):
        return len(self.ghcn) * self.sampler.times

    def __getitem__(self, idx):
        ghcn_idx = idx // self.sampler.times
        if idx % self.sampler.times == 0:  # recompute these guys only when a new ghcn_idx is seen
            self.masked_laplacian, self.x, self.non_zeros_indices = self.ghcn[ghcn_idx]
        # will be useful when figuring out the output of model at masked stations
        idx_to_mask = self.sampler.sample(self.non_zeros_indices)
        further_masked_laplacian = mask_laplacian(self.masked_laplacian, idx_to_mask)
        assert further_masked_laplacian is not self.masked_laplacian
        # mask the x as well
        masked_x = self.x.copy()
        masked_value = self.x[idx_to_mask]  # will serve as ground truth later
        masked_x[idx_to_mask] = 0.0
        return further_masked_laplacian, masked_x, idx_to_mask, masked_value

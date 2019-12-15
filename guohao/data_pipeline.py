import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ghcn_helper import build_yearbook, iterate_stations, find_all_stations, days_after
from process_laplacian import compute_cotan_laplacian, combine_cotan_mass, compute_lmax


class GHCN(Dataset):
    def __init__(self, date_start, date_end, elems_wanted,
                 times_masking, num_masked, graph_freezing, black_list):
        """
        load stuff up here for performance
        :param date_start: like "20180101"
        :param date_end: like "20181231"
        :param times_masking: graph on each day will be re-masked *times_masking* times
        :param num_masked=0.1 then 10% of stations will be masked
                         =12 then 12 stations will be masked
        :param graph_freezing: determines how often the graph has to be rebuilt.
                               if graph_freezing=7, graph will be built by intersection of stations in 7 days
        :param black_list: a set of station ids, computed externally by pre-coarsening
        """
        # I think we can simply force BATCH_SIZE == TIMES_MASKING
        # config stuff
        self.date_start = date_start
        self.date_end = date_end
        self.times_masking = times_masking
        self.num_masked = num_masked
        self.graph_freezing = graph_freezing
        # load stuff up; this can take a little while
        self.yearbook = build_yearbook(date_start, date_end, elems_wanted)
        self.df_by_date = list(iterate_stations(date_start, date_end, self.yearbook))
        # things to keep track of during training
        self.current_laplacian = None  # the M^-1 @ L product
        self.current_lmax = None
        self.omega_stations = None  # stations used to build graph; the Omega set in the frozen time period
        self.days2expire = graph_freezing  # will be reset to graph_freezing when getting zero

    def __getitem__(self, idx):
        date_str, df = self.df_by_date[idx // self.times_masking]
        # check graph expiration
        if self.days2expire == self.graph_freezing:  # brand new freezing period; everything is None
            assert self.current_laplacian is None
            assert self.omega_stations is None
            assert self.current_lmax is None
            # TODO: Build the graph
            # Very easy to make off-by-one error here; Remember that date ranges are inclusive
            self.omega_stations = find_all_stations(date_str,
                                                    days_after(date_str, self.graph_freezing-1),
                                                    self.yearbook)
            L, M = compute_cotan_laplacian(self.omega_stations)
            laplacian = combine_cotan_mass(L, M)
            lmax = compute_lmax(laplacian)

        # TODO: Some stuff in the middle

        if self.days2expire == 0:  # expired; reset everything
            self.current_laplacian = None
            self.omega_stations = None
            self.current_lmax = None
            self.days2expire = self.graph_freezing
        else:  # one day closer to expiration
            self.days2expire -= 1

        pass

    def __len__(self):
        date_range = pd.date_range(self.date_start, self.date_end)
        return len(date_range) * self.times_masking




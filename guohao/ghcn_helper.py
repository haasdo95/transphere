import pickle
import sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


path_template = "GHCN/data/processed/{}_{}_by_date.pkl"


def cartesian2spherical(x):
    """
    :param x: unit vector(s) on S^2; shaped as (3, N) or (3, )
    :return: (zenith, azimuth), in that order
    """
    zenith = np.arccos(x[2])  # [0, pi]
    azimuth = np.arctan2(x[1], x[0])  # [-pi, pi]
    return zenith, azimuth


def spherical2cartesian(zenith, azimuth):
    """
    vectorized, returned value shaped (3, N)
    :param zenith: [0, pi]
    :param azimuth: [-pi, pi]
    """
    return np.array([
        np.sin(zenith) * np.cos(azimuth),
        np.sin(zenith) * np.sin(azimuth),
        np.cos(zenith)
    ])


def geographical2spherical(latitude, longitude):
    """
    convert (latitude, longitude) to (zenith, azimuth)
    by identifying zenith=pi/2 with the Equator
                   azimuth=0 with the Prime Meridian
    :param latitude: shaped as (N, ), [-90, 90]
    :param longitude: shaped as (N, ), [-180, 180]
    """
    zenith = (90 - latitude) / 180 * np.pi
    azimuth = longitude / 180 * np.pi
    return zenith, azimuth


def chord_dist(coord_1, coord_2):
    """
    can serve as a metric for points on sphere
    :param coord_1: (lat, lon) coordinate of the first
    :param coord_2: (lat, lon) coordinate of the second
    """
    z1, a1 = geographical2spherical(*coord_1)
    z2, a2 = geographical2spherical(*coord_2)
    x1 = spherical2cartesian(z1, a1)
    x2 = spherical2cartesian(z2, a2)
    return np.linalg.norm(x1 - x2)


def pre_coarsen(df: pd.DataFrame, threshold):
    """
    this is supposed to conduct a step of VERY mild coarsening
    to remove stations that are simply too close to its neighbors
    SKETCH: (1) compute for each vertex the dist to its nearest neighbor
            (2) remove the vertex that has the smallest nearest neighbor
            (3) repeat (1) and loop on
    :param df: pandas data frame
    :param threshold:
        for threshold = 0.1 (smaller than 1 in general), we remove 10% of the stations
        for threshold = 12 (> 1 in general), we remove that many stations
    :return: coarsened data frame
    """
    df = df.copy()
    if threshold > 1:
        num_to_remove = threshold
    else:
        num_to_remove = int(len(df) * threshold)

    points = df2points(df).T
    assert points.shape[1] == 3
    for i in range(num_to_remove):
        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
        distances, _ = nn.kneighbors(points)
        distances = distances[:, 1]  # the first column is trivial
        point_to_remove = np.argmin(distances)  # the point closest to its nn
        print("{}-th removed point has nearest dist {}".format(i, distances[point_to_remove]))
        # remove point from both points and dataframe
        points = np.delete(points, axis=0, obj=point_to_remove)
        df.drop(point_to_remove, inplace=True)
        df.index = range(len(df))  # dropping will mess up the index; do so to fix it
    return df


def df2points(df):
    """
    :param df: a data frame with latitude & longitude information
    :return: point cloud shaped as (3, N)
    """
    zeniths, azimuths = geographical2spherical(df["latitude"], df["longitude"])
    return spherical2cartesian(zeniths, azimuths)


def build_yearbook(date_start, date_end, elems_wanted):
    """
    [date_left, date_right], inclusive
    :param date_start: strings like "20180101"
    :param date_end: strings like "20180101"
    :param elems_wanted: a list like ["TMAX", "TMIN"]
    :return: a lookup table of {year: df_by_date}
    """
    # figure out what years are need
    date_range = pd.date_range(date_start, date_end)
    yearbook = {d.year: None for d in date_range}
    # and load'em up
    for year in yearbook:
        path = path_template.format(year, "_".join(elems_wanted))
        with open(path, "rb") as f:
            df_by_date = pickle.load(f)
        yearbook[year] = df_by_date
    return yearbook


def find_persistent_stations(date_start, date_end, yearbook):
    """
    [date_left, date_right], inclusive
    :param date_start: strings like "20180101"
    :param date_end: strings like "20180101"
    :param yearbook: a lookup table of {year: df_by_date}
    :return:
    """
    persistent_stations = None
    date_range = pd.date_range(date_start, date_end)
    for date in date_range:
        df_by_date = yearbook[date.year]
        df = df_by_date.get_group(date.strftime("%Y%m%d"))  # df at date
        if persistent_stations is None:  # first loop
            persistent_stations = df[["station_id", "latitude", "longitude"]]
        else:
            persistent_stations = persistent_stations.merge(
                df["station_id"], on="station_id", how="inner"
            )
    # remove dup in terms of latitude and longitude
    print("Found {} persistent stations in date range {}-{}".format(len(persistent_stations), date_start, date_end))
    persistent_stations.drop_duplicates(subset=["latitude", "longitude"], keep="last")
    print("After removing dups: ", len(persistent_stations))
    return persistent_stations


def find_all_stations(date_start, date_end, yearbook):
    """
    [date_left, date_right], inclusive
    :param date_start: strings like "20180101"
    :param date_end: strings like "20180101"
    :param yearbook: a lookup table of {year: df_by_date}
    """
    all_stations = None
    date_range = pd.date_range(date_start, date_end)
    for date in date_range:
        df_by_date = yearbook[date.year]
        df = df_by_date.get_group(date.strftime("%Y%m%d"))  # df at date
        if all_stations is None:  # first loop
            all_stations = df[["station_id", "latitude", "longitude"]]
        else:  # sql-style union
            all_stations = pd.concat(
                [all_stations, df[["station_id", "latitude", "longitude"]]], ignore_index=True
            ).drop_duplicates().reset_index(drop=True)
    print("Found {} stations in total in date range {}-{}".format(len(all_stations), date_start, date_end))
    all_stations.drop_duplicates(subset=["latitude", "longitude"], keep="last")
    print("After removing dups: ", len(all_stations))
    return all_stations

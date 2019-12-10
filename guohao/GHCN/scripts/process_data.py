import pandas as pd
import pickle
import numpy as np
import os.path


def build_station_lookup_table():
    """
    glean information (e.g. location) about stations
    """
    print("---building lookup table---")
    lookup_table = {}
    with open("doc/ghcnd-stations.txt") as f:
        for line in f:
            fields = line.split()
            station_id = fields[0]
            if station_id in lookup_table:
                raise Exception("Reappearing: {}".format(station_id))
            lookup_table[station_id] = {
                "latitude": float(fields[1]),  # from 0° at the Equator to +90° northward and -90 southward
                "longitude": float(fields[2]),  # from 0° at the Prime Meridian to +180° eastward and −180° westward
            }
        return lookup_table


def generate_filtered_table(year, elems_wanted, lookup_table):
    """
    :param year: e.g. "2018"
    :param elems_wanted: an iterable of elements interested
    :param lookup_table:
    """
    print("---generating singleton tables with: ", elems_wanted, "---")
    paths_to_filtered = ["data/preprocessed/" + year + "_" + ew + ".csv" for ew in elems_wanted]
    # check if the files are already there
    # if so, no-op

    files = {ew: open(path_to_filtered, "w") for ew, path_to_filtered in zip(elems_wanted, paths_to_filtered)}
    for ew, wf in files.items():  # write header
        wf.write("station_id,date,latitude,longitude,{}\n".format(ew))
    with open("data/raw/" + year + ".csv") as f:
        for idx, line in enumerate(f):
            fields = line.split(",")
            station_id = fields[0]
            date = fields[1]
            elem_type = fields[2]
            if elem_type not in elems_wanted:  # skip if not interested
                continue
            elem_val = fields[3]
            if elem_type in ("TMAX", "TMIN"):
                elem_val = float(elem_val) / 10  # degree celsius

            if station_id not in lookup_table:
                raise Exception("Rogue Station: {}".format(station_id))
            latitude, longitude = lookup_table[station_id]["latitude"], lookup_table[station_id]["longitude"]
            write_line = "{},{},{},{},{}\n".format(
                station_id, date, latitude, longitude, elem_val
            )
            files[elem_type].write(write_line)
            if idx % 1000000 == 0:
                print(write_line)
    for _, wf in files.items():
        wf.flush()
        wf.close()
    print("---finished generating singleton tables---")


def dtypes(elems_wanted):
    dt = {'station_id': str, "date": str, "latitude": np.float, "longitude": np.float}
    for ew in elems_wanted:
        dt[ew] = np.float
    return dt


def inner_join(year, elems_wanted):
    """
    merge singleton tables
    :param year: e.g. "2018"
    :param elems_wanted:
    :return:
    """
    print("---merging singleton tables: ", elems_wanted, "---")
    assert type(year) is str
    dest_path = "data/preprocessed/" + year + "_" + "_".join(elems_wanted) + ".csv"
    if os.path.isfile(dest_path):
        print("---using cache inner joined tables---")
        joined = pd.read_csv(dest_path, dtype=dtypes(elems_wanted))
        return joined
    paths_to_filtered = ["data/preprocessed/" + year + "_" + ew + ".csv" for ew in elems_wanted]
    dfs = [pd.read_csv(path_to_filtered, dtype=dtypes([ew]))  # read in singletons
           for ew, path_to_filtered in zip(elems_wanted, paths_to_filtered)]
    joined, the_rest = dfs[0], dfs[1:]
    for ew, df in zip(elems_wanted[1:], the_rest):
        df = df[["station_id", "date", ew]]  # only keep useful sub-dataframe
        joined = pd.merge(joined, df, how="inner", on=["station_id", "date"])
    joined.to_csv(dest_path, index=False)
    return joined


def group_by_date(year, elems_wanted):
    """
    group by date before or after inner join
    :param year: e.g. "2018"
    :param elems_wanted: a list, e.g. ["TMAX", "TMIN"]
    :return:
    """
    print("---group by date---")
    assert type(year) is str
    elems_wanted_str = "_".join(elems_wanted)
    dest_path = "data/processed/" + year + "_" + elems_wanted_str + "_by_date.pkl"
    if os.path.isfile(dest_path):
        print("---using cached group by date---")
        return pickle.load(open(dest_path, "rb"))

    path_to_filtered = "data/preprocessed/" + year + "_" + elems_wanted_str + ".csv"
    df = pd.read_csv(path_to_filtered, dtype=dtypes(elems_wanted_str))
    df_groupby_date = df.groupby("date")
    with open(dest_path, "wb") as wf:
        pickle.dump(df_groupby_date, wf)
    return df_groupby_date


if __name__ == '__main__':
    station_lookup = build_station_lookup_table()
    generate_filtered_table("2018", ["TMAX", "TMIN"], station_lookup)
    inner_join("2018", ["TMAX", "TMIN"])
    df_by_date = group_by_date("2018", ["TMAX", "TMIN"])

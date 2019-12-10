"""
This files contains the functionality to:
(1) draw a sphere
(2) scatter plot on a sphere
(3) heat map on a sphere
"""
import numpy as np
import plotly.graph_objs as go


class Visualization:
    def __init__(self, zenith_res, azimuth_res):
        """
        :param zenith_res: resolution of zenith angle
        :param azimuth_res: resolution of azimuth angle
        """
        self.zenith_res = zenith_res
        self.azimuth_res = azimuth_res
        self.base_fig = self._build_mesh()
        self.points = None

    def _build_mesh(self):
        # note that we may have to remove both end points later when computing
        zeniths = np.linspace(0, np.pi, self.zenith_res)
        azimuths = np.linspace(-np.pi, np.pi, self.azimuth_res)
        # z_grid is zeniths being vertically stacked for azimuth_res times
        # a_grid is azimuths.T being horizontally stacked for zenith_res times
        # shape of both: (azimuth_res, zenith_res)
        z_grid, a_grid = np.meshgrid(zeniths, azimuths)
        x = np.sin(z_grid) * np.cos(a_grid)
        y = np.sin(z_grid) * np.sin(a_grid)
        z = np.cos(z_grid)
        data = [go.Surface(
            x=x, y=y, z=z,
            surfacecolor=np.zeros_like(z),
            colorscale="Greys",
            colorbar=dict(
                title="Surface",
            ),
        )]
        return go.Figure(data=data)

    def _update_surface_color(self, color_scheme):
        fig = self.base_fig
        surface = fig.data[0]
        mesh_shape = surface['z'].shape
        assert color_scheme.shape == mesh_shape
        cmin = np.min(color_scheme)
        cmax = np.max(color_scheme)
        surface.surfacecolor = color_scheme
        surface.cmin = cmin
        surface.cmax = cmax

    def update_surface_color(self, f):
        """
        :param f: a function of zenith and azimuth; expected to be vectorized properly
        """
        zeniths = np.linspace(0, np.pi, self.zenith_res)
        azimuths = np.linspace(-np.pi, np.pi, self.azimuth_res)
        z_grid, a_grid = np.meshgrid(zeniths, azimuths)
        color_scheme = f(z_grid, a_grid)
        self._update_surface_color(color_scheme)

    def scatter(self, points):
        """
        scatter plot on sphere
        :param points: numpy array shaped (3, N)
        """
        assert points.shape[0] == 3
        self.points = points
        self.remove_scatter_plot()
        x, y, z = zip(*points.T)
        self.base_fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=np.zeros(points.shape[1]),
                colorscale="RdBu",
                colorbar=dict(
                    title="Samples",
                    x=-0.2
                ),
            )
        ))

    def remove_scatter_plot(self):
        if len(self.base_fig.data) != 1:
            print("Removing previous scatter plot")
            self.base_fig.data = self.base_fig.data[:1]

    def update_scatter_color(self, c):
        """
        :param c: shaped as (N, )
        """
        if self.points is None:
            raise Exception("No Scatter Plot Detected")
        cmin_scatter = np.min(c)
        cmax_scatter = np.max(c)
        # grab the fig
        fig = self.base_fig
        scatter = fig.data[1]
        scatter.marker["cmin"] = cmin_scatter
        scatter.marker["cmax"] = cmax_scatter
        scatter.marker["color"] = c

    def show(self):
        self.base_fig.show()

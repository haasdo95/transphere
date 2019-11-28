Main goal: solve the problem of uneven sampling on the manifolds

How to solve it: setting vertex weights

We need to show empirically (weather data) that show we do it now is better than what we
did so far.

we need to show that with this method it does not depend how I sample the manifold
(kind of transferability with respect to sampling)
Problem: how to measure it?


Guohao: interpolation, with different graph constructions. Comparison.
Test with real data.

transferability to different samples but with same sample size!

Experiments:
* interpolation on weather data. No time, mask stations. Interpolate Tmax
* supervised on non-uniform (- extreme events from climate simulations,
  10M pixels - icosahedral vs equiangular)
* aerial images (purpose is to go beyond the sphere)

Graphs:
* hodge/lumped FEM (full FEM only for theoretical discussions)
* same, but without vertex weight
* B&N
* UMAP
* variable bandwidth

Test:
* highly non uniform sampling (spectrum). depending on the reults, we decide which experiment to do first

good eigenvalues necessary for transferability between sampings
good eigenvectors necessary for equivariance

Martino:
* lumped FEM on GHCN data: check spectrum. to do it: deepsphere_v2_paper/figures/example_ghcn_graph.py
* create PyGSP class with the graph construction (store in G.L the non symmetric one)
* 

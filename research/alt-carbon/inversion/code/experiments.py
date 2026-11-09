"""Synthetic stress tests for magnetic inversion (Task 2C).

E1  Remanence: scalar inversion of data from a remanent body -> negative chi?
E2  Control: no remanence; do negatives still appear (unbounded / regional error)?
E3  Overfitting: beta sweep, train vs hold-out-line misfit, model error.
E4  Memory/time scaling of the dense sensitivity matrix.
"""
import json, time, sys
import numpy as np
import discretize
from simpeg import maps, data as sdata, data_misfit, regularization, optimization, \
    inverse_problem, inversion, directives
from simpeg.potential_fields import magnetics as mag

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(42)
B0, INC, DEC = 52000.0, 75.0, 10.0   # roughly Finland-like field

def unit(inc, dec):
    inc, dec = np.deg2rad(inc), np.deg2rad(dec)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])
# SimPEG convention: x=east, y=north, z=up; positive inclination points down.

def build(nx=24, ny=24, nz=12, h=25.0):
    mesh = discretize.TensorMesh([[(h, nx)], [(h, ny)], [(h, nz)]], origin="CCN")
    xs = mesh.cell_centers_x; ys = mesh.cell_centers_y
    X, Y = np.meshgrid(xs, ys)
    locs = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 30.0)]
    line_id = np.repeat(np.arange(len(ys)), len(xs))   # flight lines along x
    return mesh, locs, line_id

def survey_for(locs):
    rx = mag.receivers.Point(locs, components="tmi")
    src = mag.sources.UniformBackgroundField([rx], amplitude=B0, inclination=INC, declination=DEC)
    return mag.survey.Survey(src)

def prism(mesh, x0, x1, y0, y1, z0, z1):
    c = mesh.cell_centers
    return (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z0) & (c[:, 2] < z1)

def true_models(mesh, remanent=True):
    nC = mesh.n_cells
    chi = np.zeros(nC)
    A = prism(mesh, -200, -75, -100, 100, -175, -50)
    B = prism(mesh, 75, 175, -100, 100, -150, -50)
    chi[A] = 0.05; chi[B] = 0.03
    b = unit(INC, DEC)
    M = np.outer(chi, b)                    # induced part (effective susceptibility vector)
    if remanent:
        Q = 3.0
        r = unit(-60.0, 190.0)              # reversed remanence
        M[B] += Q * 0.03 * r
    return chi, M, A, B

def forward(mesh, locs, model, vector):
    nC = mesh.n_cells
    sim = mag.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey_for(locs),
        chiMap=maps.IdentityMap(nP=3 * nC if vector else nC),
        model_type="vector" if vector else "scalar",
        store_sensitivities="ram", engine="choclo")
    m = model.T.ravel() if vector else model   # [mx..., my..., mz...]
    return sim, sim.dpred(m)

def invert(mesh, locs, dobs, std, beta, lower=-np.inf, max_iter=15):
    nC = mesh.n_cells
    sim = mag.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey_for(locs), chiMap=maps.IdentityMap(nP=nC),
        store_sensitivities="ram", engine="choclo")
    dobj = sdata.Data(sim.survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=dobj, simulation=sim)
    reg = regularization.WeightedLeastSquares(mesh, reference_model=np.zeros(nC),
                                              alpha_s=1e-4, alpha_x=1, alpha_y=1, alpha_z=1)
    opt = optimization.ProjectedGNCG(maxIter=max_iter, lower=lower, upper=np.inf,
                                     maxIterLS=20, cg_maxiter=50, cg_rtol=1e-4)
    opt.remember("xc")
    prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=beta)
    dirs = [directives.UpdateSensitivityWeights(every_iteration=False),
            directives.UpdatePreconditioner()]
    inv = inversion.BaseInversion(prob, directiveList=dirs)
    m = inv.run(np.full(nC, 1e-4))
    return m, sim

def metrics(m, chi_true, sim_all, d_all_clean, std_all, train, test):
    pred = sim_all.dpred(m)
    r = (pred - d_all_clean) / std_all
    return dict(
        phi_train_per_datum=float(np.mean(((pred[train] - D_OBS[train]) / std_all[train]) ** 2)),
        rms_test_vs_obs=float(np.sqrt(np.mean(((pred[test] - D_OBS[test]) / std_all[test]) ** 2))),
        rms_test_vs_clean_nT=float(np.sqrt(np.mean((pred[test] - d_all_clean[test]) ** 2))),
        model_rmse=float(np.sqrt(np.mean((m - chi_true) ** 2))),
        min_chi=float(m.min()), max_chi=float(m.max()),
        frac_neg_cells=float(np.mean(m < -0.002)),
    )

if __name__ == "__main__":
    res = {}
    mesh, locs, line_id = build()
    train = line_id % 4 != 0; test = ~train
    nC = mesh.n_cells
    print("cells", nC, "data", len(locs), "train", train.sum(), "test", test.sum(), flush=True)
    sim_scalar_all = mag.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey_for(locs), chiMap=maps.IdentityMap(nP=nC),
        store_sensitivities="ram", engine="choclo")

    models = {}
    for case, rem in [("remanent", True), ("induced", False)]:
        chi, M, A, B = true_models(mesh, rem)
        _, d_clean = forward(mesh, locs, M, vector=True)
        std = 0.02 * np.abs(d_clean) + 2.0
        d_obs = d_clean + rng.normal(0, std)
        models[case] = (chi, d_clean, d_obs, std, A, B)
    np.save(f"{OUT}/mesh_shape.npy", np.array(mesh.shape_cells))

    def run(case, beta, lower=-np.inf, offset=0.0, tag=None):
        global D_OBS
        chi, d_clean, d_obs, std, A, B = models[case]
        d_use = d_obs + offset
        D_OBS = d_use
        t = time.time()
        m, _ = invert(mesh, locs[train], d_use[train], std[train], beta, lower)
        out = metrics(m, chi, sim_scalar_all, d_clean + offset, std, train, test)
        out.update(case=case, beta=beta, lower=None if np.isinf(lower) else lower,
                   offset_nT=offset if np.isscalar(offset) else "trend",
                   seconds=round(time.time() - t, 1),
                   mean_chi_bodyA=float(m[A].mean()), mean_chi_bodyB=float(m[B].mean()))
        key = tag or f"{case}_b{beta:g}_{'pos' if out['lower'] == 0 else 'free'}"
        np.save(f"{OUT}/model_{key}.npy", m)
        res[key] = out
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
        return m

    BETA0 = 1e3
    # E1/E2: remanent vs induced, bounded vs free
    for case in ["remanent", "induced"]:
        for lower in [-np.inf, 0.0]:
            run(case, BETA0, lower)
    # E2b: regional-field error (uncorrected +/-) on induced data, free
    X = locs[:, 0]
    run("induced", BETA0, -np.inf, offset=-25.0, tag="induced_offset-25_free")
    run("induced", BETA0, -np.inf, offset=0.1 * X, tag="induced_trend_free")
    # E3: beta sweep on induced data, free
    for beta in [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        if beta == BETA0:
            continue
        run("induced", beta, -np.inf)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

    # E4: scaling of dense sensitivity
    scal = []
    for n in [12, 18, 24, 30, 36]:
        me, lo, _ = build(nx=n, ny=n, nz=n // 2)
        t = time.time()
        s = mag.simulation.Simulation3DIntegral(mesh=me, survey=survey_for(lo),
                                                chiMap=maps.IdentityMap(nP=me.n_cells),
                                                store_sensitivities="ram", engine="choclo")
        G = s.G
        scal.append(dict(cells=me.n_cells, data=len(lo), G_MB=G.nbytes / 1e6, seconds=time.time() - t))
        print(scal[-1], flush=True)
        del G, s
    json.dump(scal, open(f"{OUT}/scaling.json", "w"), indent=1)
    print("done")

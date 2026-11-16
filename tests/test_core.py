import numpy as np
import pytest
from courtvision.camera import calibrate_focal, pose_from_h
from courtvision.court import CourtRegistrar
from courtvision.geometry import apply_h, h_from_4pts
from courtvision.jumps import detect_jumps
from courtvision.lens import LensModel, estimate_plumbline
from courtvision.stats import analyze, cameras_from_h
from courtvision.synth import SynthCamera, random_camera, render_scene, simulate_detections, simulate_players
from courtvision.templates import TEMPLATES, get_template
from courtvision.tracking import Track, filter_and_stitch, kalman_rts
from courtvision.validate import court_error


def cam0(lam=0.0):
    return SynthCamera((1280, 720), 1000.0, np.array([9.0, -16.0, 9.0]), np.array([9.0, 4.5, 0.0]), lam)


def test_templates_dimensions():
    assert (TEMPLATES["indoor"].length, TEMPLATES["indoor"].width) == (18.0, 9.0)
    assert (TEMPLATES["beach"].length, TEMPLATES["beach"].width) == (16.0, 8.0)
    assert len(TEMPLATES["indoor"].segments) == 7
    for t in TEMPLATES.values():
        pts, idx = t.sample(0.5)
        assert len(pts) == len(idx) and pts[:, 0].max() <= t.length + 1e-9


def test_batched_homography_exact():
    rng = np.random.default_rng(0)
    src = np.array([[0, 0], [18, 0], [18, 9], [0, 9.0]])
    dst = src * 30 + rng.normal(0, 20, (5, 4, 2)) + 200
    H = h_from_4pts(np.broadcast_to(src, dst.shape), dst)
    for i in range(5):
        assert np.allclose(apply_h(H[i], src), dst[i], atol=1e-6)


def test_lens_roundtrip_and_maps():
    m = LensModel(-0.2, (1280, 720))
    p = np.random.default_rng(0).uniform([0, 0], [1280, 720], (200, 2))
    assert np.allclose(m.distort_points(m.undistort_points(p)), p, atol=1e-6)


def test_focal_and_pose_recovered_from_homography():
    cam = cam0()
    f, resid, obs = calibrate_focal(cam.H, cam.size)
    assert abs(f / cam.f - 1) < 0.01 and resid < 1e-6 and obs > 10
    est = pose_from_h(cam.H, f, cam.size)
    assert np.allclose(est.C, cam.C, atol=0.15)
    assert np.allclose(est.project([[5, 3, 0.7]]), cam.project([[5, 3, 0.7]]), atol=1.5)


@pytest.mark.parametrize("court", ["indoor", "beach"])
def test_court_registration_clean(court):
    tpl = get_template(court)
    cam = cam0()
    img = render_scene(cam, tpl, np.random.default_rng(1), clutter=False)
    reg = CourtRegistrar(tpl).detect(img)
    assert reg is not None and reg.ok
    assert court_error(reg.H, cam, tpl)[1] < 0.05           # < 5 cm on the ground


def test_court_registration_with_clutter_and_occluders_then_track():
    tpl = get_template("indoor")
    cam = cam0()
    rng = np.random.default_rng(2)
    gt = simulate_players(tpl, rng, duration=2.0)
    img = render_scene(cam, tpl, rng, clutter=True, players_xyz=np.c_[gt["xy"][:, 0], gt["z"][:, 0]])
    r = CourtRegistrar(tpl)
    reg = r.update(img)
    assert reg.ok and court_error(reg.H, cam, tpl)[1] < 0.08
    reg2 = r.update(img)
    assert reg2.source == "track" and court_error(reg2.H, cam, tpl)[1] < 0.08
    assert reg2.H_samples.shape == (16, 3, 3)


def test_plumbline_recovers_distortion():
    tpl = get_template("indoor")
    cam = cam0(lam=-0.2)
    img = render_scene(cam, tpl, np.random.default_rng(3), clutter=True)
    m = estimate_plumbline(img)
    assert m.significant and abs(m.lam - (-0.2)) < 0.04
    m0 = estimate_plumbline(render_scene(cam0(), tpl, np.random.default_rng(3), clutter=True))
    assert not m0.significant


def test_kalman_bridges_gaps():
    t = np.arange(100) / 25.0
    z = np.c_[2.0 * t, 1.0 + 0 * t] + np.random.default_rng(0).normal(0, 0.05, (100, 2))
    z[40:55] = np.nan
    pos, vel = kalman_rts(z, 1 / 25.0)
    assert np.abs(pos[47] - [2.0 * t[47], 1.0]).max() < 0.2
    assert abs(np.median(vel[:, 0]) - 2.0) < 0.2


def test_jump_height_from_geometry():
    cam = cam0()
    fps, T = 25.0, 100
    xy = np.tile([[12.0, 4.0]], (T, 1)); z = np.zeros(T)
    h_true, n = 0.5, int(round(np.sqrt(8 * 0.5 / 9.81) * fps))
    tau = np.arange(1, n) / n
    z[40:40 + n - 1] = 4 * h_true * tau * (1 - tau)
    uv = cam.project(np.c_[xy, z]) + np.random.default_rng(0).normal(0, 0.5, (T, 2))
    H = np.tile(cam.H, (T, 1, 1))
    P, f, _ = cameras_from_h(H, cam.size)
    air, ground, zs, jumps = detect_jumps(uv, np.linalg.inv(H), P, fps)
    assert len(jumps) == 1
    assert abs(jumps[0]["height_m"] - h_true) < 0.08
    assert np.abs(ground[45] - [12.0, 4.0]).max() < 0.15      # airborne frames are not projected


def test_stats_end_to_end_synthetic():
    tpl = get_template("indoor")
    cam = cam0()
    rng = np.random.default_rng(5)
    gt = simulate_players(tpl, rng, duration=10.0)
    tracks = simulate_detections(cam, gt, rng, noise_px=1.0)
    T = gt["xy"].shape[1]
    H = np.tile(cam.H, (T, 1, 1)); Hs = np.tile(cam.H, (T, 8, 1, 1))
    res, info = analyze(tracks, H, Hs, 25.0, tpl, cam.size)
    assert len(res) == 12
    err = []
    for p, r in enumerate(res):
        d = np.linalg.norm(np.diff(gt["xy"][p], axis=0), axis=1).sum()
        err.append(abs(r.stats["distance_m"] - d) / max(d, 1.0))
        assert r.stats["distance_m_lo"] <= r.stats["distance_m"] <= r.stats["distance_m_hi"]
        assert 0 <= r.stats["front_row_frac"] <= 1
    assert np.median(err) < 0.12


@pytest.mark.parametrize("court", ["padel", "football5"])
def test_other_sport_templates_register(court):
    tpl = get_template(court)
    rng = np.random.default_rng(0)
    cam = random_camera(tpl, rng)
    reg = CourtRegistrar(tpl).detect(render_scene(cam, tpl, rng, clutter=False))
    assert reg is not None and reg.ok and court_error(reg.H, cam, tpl)[1] < 0.1


def test_occluded_legs_are_not_a_jump():
    """Box bottom rises but the box gets shorter -> occlusion, not a jump."""
    cam = cam0()
    fps, T = 25.0, 100
    xy = np.tile([[12.0, 4.0]], (T, 1)); z = np.zeros(T)
    n = 16; tau = np.arange(1, n) / n
    z[40:40 + n - 1] = 4 * 0.5 * tau * (1 - tau)
    uv = cam.project(np.c_[xy, z])
    H = np.tile(cam.H, (T, 1, 1))
    P, _, _ = cameras_from_h(H, cam.size)
    head = cam.project(np.c_[xy, np.full(T, 1.8)])
    box_h = head[:, 1] * 0 + np.abs(head[:, 1] - uv[:, 1])           # head fixed, feet "rise"
    assert len(detect_jumps(uv, np.linalg.inv(H), P, fps, box_h=box_h)[3]) == 0
    assert len(detect_jumps(uv, np.linalg.inv(H), P, fps, top_uv=head - [0, 0])[3]) == 0     # head did not rise
    assert len(detect_jumps(uv, np.linalg.inv(H), P, fps, box_h=np.full(T, 100.0))[3]) == 1
    head_jump = cam.project(np.c_[xy, z + 1.8])
    assert len(detect_jumps(uv, np.linalg.inv(H), P, fps, top_uv=head_jump)[3]) == 1         # whole body rose


def _walk(tid, frames, start, vel, rng):
    """Track walking in court metres; identity homography, so foot pixels are metres."""
    frames = np.asarray(frames)
    xy = np.asarray(start) + np.outer(frames, vel) + rng.normal(0, 0.03, (len(frames), 2))
    return Track(tid, frames, xy, np.zeros((len(frames), 4)), np.zeros((len(frames), 3)))


def test_stitch_id_handover_with_overlap():
    """ByteTrack starts a new id while still emitting the old one: stitch and trim the old tail."""
    rng = np.random.default_rng(0)
    tpl, fps, T = get_template("indoor"), 25.0, 300
    Hinv = np.tile(np.eye(3), (T, 1, 1))
    v = (0.02, 0.0)
    a = _walk(1, range(0, 160), (2.0, 4.0), v, rng)            # old id, emitted until frame 159
    b = _walk(2, range(150, 300), (2.0, 4.0), v, rng)          # new id from frame 150: 10 frames overlap
    other = _walk(3, range(0, 300), (6.0, 1.0), v, rng)        # a different player, far away
    out = {t.tid: t for t in filter_and_stitch([a, b, other], Hinv, fps, tpl)}
    assert set(out) == {1, 3}
    m = out[1]
    assert len(m.frames) == 300 and np.all(np.diff(m.frames) == 1)


def test_stitch_rejects_overlap_with_other_player_or_in_a_cluster():
    rng = np.random.default_rng(1)
    tpl, fps, T = get_template("indoor"), 25.0, 300
    Hinv = np.tile(np.eye(3), (T, 1, 1))
    v = (0.02, 0.0)
    # two different players 2 m apart, one id ends while the other starts: must not merge
    a = _walk(1, range(0, 160), (2.0, 4.0), v, rng)
    b = _walk(2, range(150, 300), (2.0, 6.0), v, rng)
    assert len(filter_and_stitch([a, b], Hinv, fps, tpl)) == 2
    # same-spot handover, but a third player stands 0.5 m from the new id: ambiguous, keep apart
    b = _walk(2, range(150, 300), (2.0, 4.0), v, rng)
    c = _walk(3, range(0, 300), (2.0, 4.5), v, rng)
    assert len(filter_and_stitch([a, b, c], Hinv, fps, tpl)) == 3


@pytest.mark.parametrize("seed", [2, 7])
def test_online_matches_ground_truth_with_id_switches(seed):
    """Frames pushed one by one; ByteTrack gives a new id after every occlusion gap. The online
    analyzer must keep one slot per player and report distances close to the truth, with a fixed delay."""
    from types import SimpleNamespace
    from courtvision.online import OnlineAnalyzer, OnlineConfig
    tpl, fps = get_template("indoor"), 25.0
    cam = cam0()
    rng = np.random.default_rng(seed)
    gt = simulate_players(tpl, rng, duration=12.0)
    tracks = simulate_detections(cam, gt, rng, noise_px=1.0)
    T = gt["xy"].shape[1]
    kit = {0: np.array([30.0, 128, 128]), 1: np.array([220.0, 128, 128])}
    dets, next_id = [[] for _ in range(T)], 1000
    for p, tr in enumerate(tracks):
        side = int(np.median(gt["xy"][p][:, 0]) > tpl.net_x)
        tid = p
        for i, f in enumerate(tr.frames):
            if i and f - tr.frames[i - 1] > 1:
                tid, next_id = next_id, next_id + 1            # new id after each gap
            dets[f].append({"tid": tid, "foot": tr.feet[i], "box": tr.boxes[i], "conf": 0.8,
                            "color": kit[side] + rng.normal(0, 3, 3)})
    reg = SimpleNamespace(H=cam.H, H_samples=np.tile(cam.H, (8, 1, 1)), ok=True)
    on = OnlineAnalyzer(tpl, fps, cam.size, OnlineConfig(lag_s=2.0))
    lags = []
    for t in range(T):
        out = on.push(reg, dets[t])
        if out:
            lags.append(t + 1 - on.done)
    on.flush()
    assert len(on.slots) == 12 and len(on.frames_out) == T
    assert min(lags) >= 50 and max(lags) <= 50 + on.hop          # 2 s delay, finalised in hops
    pos = np.full((T, 12, 2), np.nan)
    for f, fr in enumerate(on.frames_out):
        for sid, v in fr.items():
            pos[f, sid] = v[0]
    st = {s["sid"]: s for s in on.stats()}
    err, same = [], []
    for sid in range(12):
        d = np.linalg.norm(pos[:, sid][:, None] - gt["xy"].transpose(1, 0, 2), axis=2)   # (T, players)
        p = int(np.nanargmin(np.nanmedian(d, 0)))
        m = np.isfinite(d[:, p])
        same.append((d[m, p] < 0.5).mean())
        truth = np.linalg.norm(np.diff(gt["xy"][p], axis=0), axis=1).sum()
        err.append(abs(st[sid]["distance_m"] - truth) / max(truth, 1.0))
    # same-kit players crossing while hidden can be swapped (offline stitching swaps them too):
    # allow one swapped pair
    assert sum(x > 0.95 for x in same) >= 10 and min(same) > 0.8
    assert np.median(err) < 0.05

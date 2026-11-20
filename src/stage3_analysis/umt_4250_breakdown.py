"""#4250 breakdown in pure PE / mixed / content terms (20/11/26). CPU, local.
(1) Down-class (43) DFA per layer split by bucket, signed for / against, R vs C (all-174 trained-threshold DFA).
(2) #4250's full L5 / L7 linear attribution summed by bucket. (3) Perturbation behaviour by bucket and per input.
Buckets: dictionary-wide rule (umt_layer_pe_characterise features_L{L}.csv; cut blank >= 0.8 x real).
Out: outputs/analysis/umt_4250_breakdown/*.csv + *.png. Usage: uv run python src/stage3_analysis/umt_4250_breakdown.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
A = ROOT / "outputs/analysis"
CFG = {"layers": (5, 7, 9), "cls": 43, "target": 4250, "active": 0.5, "pc": A / "umt_layer_pe_characterise",
       "dfa": A / "dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l{}_job7ep_k64.parquet",
       "out": A / "umt_4250_breakdown", "buckets": ("pure PE", "mixed", "content"), "conds": ("R", "C")}   # VM: ("R", "C1")
NAME = {"pe": "pure PE", "mixed": "mixed", "content": "content", "dead": "dead"}


def buckets(L, cfg):
    """(F,) bucket name per latent at layer L (dictionary rule) and the per-latent features table."""
    f = pd.read_csv(cfg["pc"] / f"features_L{L}.csv").set_index("latent")
    return f.cls.map(NAME).to_numpy(), f


def down_dfa(L, cond, cfg):
    """(F,) mean per-clip signed DFA of the down logit at layer L over down clips (R-correct), condition R or C."""
    d = pd.read_parquet(str(cfg["dfa"]).format(L), columns=["class_id", f"signed_vec_{cond}"])
    d = d[d.class_id == cfg["cls"]]
    return np.stack(d[f"signed_vec_{cond}"].to_numpy()).astype(np.float64).mean(0), len(d)


def dfa_table(cfg):
    """Per layer x condition x bucket: DFA for (sum of positive latents) and against (sum of negative), mean per clip,
    and each as % of that layer-condition's total for / against."""
    rows = []
    for L in cfg["layers"]:
        b, _ = buckets(L, cfg)
        for cond in cfg["conds"]:
            v, n = down_dfa(L, cond, cfg)
            pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
            for k in cfg["buckets"]:
                m = b == k
                rows.append(dict(layer=L, cond=cond, bucket=k, n_clips=n, dfa_for=pos[m].sum(), dfa_against=neg[m].sum(),
                                 pct_for=100 * pos[m].sum() / pos.sum(), pct_against=100 * neg[m].sum() / neg.sum()))
    return pd.DataFrame(rows)


def target_in_decision(cfg):
    """#4250's mean down DFA at L9 (R): value, share of all positive DFA, rank among 6,144 latents."""
    v, n = down_dfa(9, "R", cfg)
    t = v[cfg["target"]]
    return pd.DataFrame([dict(latent=cfg["target"], dfa=t, pct_of_positive=100 * t / np.clip(v, 0, None).sum(),
                              rank=int((v > t).sum()) + 1, n_clips=n, bucket=buckets(9, cfg)[0][cfg["target"]])])


def attr_by_bucket(cfg):
    """#4250's linear attribution from every L5 / L7 latent (311 down clips), summed by bucket: for (+) and
    against (-), % of #4250's activation."""
    from umt_4250_inputs_by_layer import attr_by_frame
    rows = []
    for L in (5, 7):
        a, b = attr_by_frame(L).sum(0), buckets(L, cfg)[0]
        for k in cfg["buckets"]:
            m = b == k
            rows.append(dict(layer=L, bucket=k, for_4250=a[m & (a > 0)].sum(), against_4250=a[m & (a < 0)].sum(), net=a[m].sum()))
    return pd.DataFrame(rows)


def behaviour(L, cfg):
    """Per latent at L: blank / real, frozen-PE kept on blank, order index (shape under a reversed video), bucket.
    Clip-total C / R and Rev / R over all 696 clips were ~1.00 for every bucket (no order signal in totals): dropped."""
    b, f = buckets(L, cfg)
    return pd.DataFrame({"latent": np.arange(len(b)), "layer": L, "bucket": b, "active": f.fire_rate.values >= cfg["active"],
                         "blank_real": f.blank_real.values, "frozen_kept": f.frozen_kept.values, "order_index": f.order_index.values})


def down_rev_l7():
    """(F,) L7 Rev / R on the 311 down clips (clip totals, baseline arm of the L5 raw run): order use on down video."""
    D = A / "umt_clock_downstream_L5_raw"
    idx = np.where(pd.read_csv(D / "clips.csv").class_id.to_numpy() == 43)[0]
    R, Rv = (np.load(D / f"prof_L7_{k}.npy", mmap_mode="r")[idx, 0].astype(np.float32).sum(1).mean(0) for k in ("R", "Rev"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return Rv / R


def behaviour_tables(cfg):
    """(per-bucket medians over active latents at L5 / L7 / L9, the same measures for #4250's top-20 L5 / L7 inputs)."""
    B = pd.concat([behaviour(L, cfg) for L in cfg["layers"]])
    med = (B[B.active & B.bucket.isin(cfg["buckets"])].groupby(["layer", "bucket"])
           [["frozen_kept", "order_index"]].median().reset_index())
    med.loc[med.bucket == "content", "frozen_kept"] = np.nan          # silent on blank: ratio has no meaning
    inp = pd.read_csv(A / "umt_4250_inputs_by_layer/inputs_4250_layers.csv")[["layer", "feature", "attr"]]
    inp = inp.merge(B.drop(columns="active"), left_on=["layer", "feature"], right_on=["layer", "latent"]).drop(columns="latent")
    rev7 = down_rev_l7()
    inp["down_Rev_over_R"] = [rev7[f] if L == 7 else np.nan for L, f in zip(inp.layer, inp.feature)]
    return med, inp.rename(columns={"attr": "pct_of_4250_linear"})


COL = {"pure PE": "#2a78d6", "mixed": "#eb6834", "content": "#1baf7a"}


def plot_dfa(t, cfg):
    """Share of down DFA for (above 0) and against (below 0) by bucket; L5 / L7 / L9 x R / C."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5)); xs, labels = [], []
    for i, (L, cond) in enumerate([(L, c) for L in cfg["layers"] for c in cfg["conds"]]):
        x = i + i // 2 * 0.6; xs.append(x); labels.append(f"L{L} {cond}"); up = dn = 0
        for k in cfg["buckets"]:
            r = t[(t.layer == L) & (t.cond == cond) & (t.bucket == k)].iloc[0]
            ax.bar(x, r.pct_for, bottom=up, color=COL[k], edgecolor="white", label=k if i == 0 else None)
            ax.bar(x, -r.pct_against, bottom=dn, color=COL[k], edgecolor="white", alpha=0.55)
            ax.text(x, up + r.pct_for / 2, f"{r.pct_for:.0f}", ha="center", va="center", fontsize=7, color="white")
            ax.text(x, dn - r.pct_against / 2, f"{r.pct_against:.0f}", ha="center", va="center", fontsize=7)
            up, dn = up + r.pct_for, dn - r.pct_against
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs, labels); ax.set_ylabel("% of down DFA: for (up) / against (down)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.set_title(f"{cfg.get('model', 'UMT')} down decision (class 43): where the DFA sits, by group and layer (R = real, {cfg['conds'][1]} = frames shuffled)", fontsize=10)
    fig.tight_layout(); fig.savefig(cfg["out"] / "down_dfa_by_layer.png", dpi=150); plt.close(fig)


def plot_attr(a, cfg):
    """#4250's linear attribution from all L5 / L7 latents by bucket: for (solid), against (faded), net (black tick)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5)); w = 0.25
    for i, L in enumerate((5, 7)):
        for j, k in enumerate(cfg["buckets"]):
            r, x = a[(a.layer == L) & (a.bucket == k)].iloc[0], i + (j - 1) * w
            ax.bar(x, r.for_4250, w * 0.9, color=COL[k], label=k if i == 0 else None)
            ax.bar(x, r.against_4250, w * 0.9, color=COL[k], alpha=0.45)
            ax.plot([x - w * 0.4, x + w * 0.4], [r.net] * 2, color="k", lw=2)
            ax.text(x, r.net + (3 if r.net >= 0 else -6), f"net {r.net:+.0f}", ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks([0, 1], ["from L5", "from L7"]); ax.set_ylabel("% of #4250 (linear attribution)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    ax.set_title("How #4250 is built: every L5 / L7 latent, summed by bucket (311 down clips)", fontsize=10)
    fig.tight_layout(); fig.savefig(cfg["out"] / "4250_attr_by_bucket.png", dpi=150); plt.close(fig)


def plot_behaviour(med, inp, cfg):
    """Left: median order index by bucket and layer. Right: #4250's L7 inputs, PE spectrum vs Rev / R on down clips."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in cfg["buckets"]:
        m = med[med.bucket == k].sort_values("layer")
        a1.plot([f"L{L}" for L in m.layer], m.order_index, marker="o", color=COL[k], lw=2, label=k)
    a1.axhline(0, color="grey", lw=0.6); a1.set_ylabel("order index (+ = profile stays put when video reversed)")
    a1.set_title("Active latents: median order index", fontsize=10); a1.legend(frameon=False, fontsize=8)
    d = inp[inp.layer == 7]
    for _, r in d.iterrows():
        x = np.log10(max(r.blank_real, 1e-3)); a2.scatter(x, r.down_Rev_over_R, s=20 + 8 * abs(r.pct_of_4250_linear), color=COL[r.bucket],
                                                          edgecolor="k" if r.pct_of_4250_linear < 0 else "none")
        a2.annotate(f"#{int(r.feature)}", (x, r.down_Rev_over_R), fontsize=7, xytext=(4, 3), textcoords="offset points")
    a2.axhline(1, color="grey", lw=0.6, ls=":"); a2.invert_xaxis()
    a2.set_xlabel("log10(blank / real): pure PE  <-  ->  content"); a2.set_ylabel("reversed / real, on down clips")
    a2.set_title("#4250's top-20 L7 inputs: size = |% of #4250|, black edge = against", fontsize=10)
    for ax in (a1, a2): ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(cfg["out"] / "behaviour_by_bucket.png", dpi=150); plt.close(fig)


def main(cfg):
    cfg["out"].mkdir(parents=True, exist_ok=True)
    t, d9, a = dfa_table(cfg), target_in_decision(cfg), attr_by_bucket(cfg)
    med, inp = behaviour_tables(cfg)
    for n, x in (("down_dfa_by_layer", t), ("4250_in_decision", d9), ("4250_attr_by_bucket", a), ("behaviour_by_bucket", med), ("4250_inputs_behaviour", inp)):
        x.round(4).to_csv(cfg["out"] / f"{n}.csv", index=False)
    av = activation_vs_use(cfg); av.round(4).to_csv(cfg["out"] / "activation_vs_use.csv", index=False)   # 20/11
    plot_dfa(t, cfg); plot_attr(a, cfg); plot_behaviour(med, inp, cfg); plot_act_vs_use(av, cfg)



def activation_vs_use(cfg):
    """Per layer and bucket on down clips: share of activation vs share of |DFA| and of DFA for 'down'.
    Activation: L5 = PE probe's 8 down clips; L7 / L9 = 311 down clips (L5 raw run, baseline arm). DFA as dfa_table."""
    D = A / "umt_clock_downstream_L5_raw"
    idx = np.where(pd.read_csv(D / "clips.csv").class_id.to_numpy() == cfg["cls"])[0]
    act = {5: np.load(A / "umt_pe_probe/real_43.npz")["full"].astype(np.float64).sum(1).mean(0)}
    for L in (7, 9):
        act[L] = np.load(D / f"prof_L{L}_R.npy", mmap_mode="r")[idx, 0].astype(np.float64).sum(1).mean(0)
    rows = []
    for L in cfg["layers"]:
        b, v = buckets(L, cfg)[0], down_dfa(L, "R", cfg)[0]
        for k in cfg["buckets"]:
            m = b == k
            rows.append(dict(layer=L, bucket=k, pct_activation=100 * act[L][m].sum() / act[L].sum(),
                             pct_abs_dfa=100 * np.abs(v[m]).sum() / np.abs(v).sum(),
                             pct_dfa_for=100 * np.clip(v, 0, None)[m].sum() / np.clip(v, 0, None).sum()))
    return pd.DataFrame(rows)


def plot_act_vs_use(t, cfg):
    """Per bucket across layers: share of |DFA| (solid) vs share of activation (dashed), down clips."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in cfg["buckets"]:
        m = t[t.bucket == k].sort_values("layer"); x = [f"L{L}" for L in m.layer]
        ax.plot(x, m.pct_abs_dfa, color=COL[k], lw=2.2, marker="o", label=f"{k}: use (|DFA|)")
        ax.plot(x, m.pct_activation, color=COL[k], lw=1.6, ls="--", marker="o", mfc="white", label=f"{k}: activation")
    ax.set_ylabel("% of layer total, down clips"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title(f"{cfg.get('model', 'UMT')}: share of |DFA| (solid) vs share of activation (dashed) by group, down clips", fontsize=10)
    fig.tight_layout(); fig.savefig(cfg["out"] / "activation_vs_use.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(CFG)

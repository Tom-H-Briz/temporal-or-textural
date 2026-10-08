"""Text-only geometry of UMT's final VTC template embeddings (174 x 512).
Checks contrast pairs (push/pull, LtR/RtL, ...) and whether verb-based slicing is coherent."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CFG = {
    "text_emb_path": Path("notebooks/umt_ssv2_template_text_emb.npy"),  # class-id ordered
    "labels_path":   Path("data/ssv2/labels/labels.json"),
    "center":        True,   # subtract the 174-class mean embedding (after unit-norm), then re-norm
    "out_dir":       Path("outputs/analysis/umt_text_geometry_centered"),
    # (class_a, class_b, contrast) — class ids from labels.json
    "pairs": [(93, 94, "push LtR vs RtL"), (86, 87, "pull LtR vs RtL"),
              (93, 86, "push vs pull, LtR"), (94, 87, "push vs pull, RtL"),
              (93, 87, "push LtR vs pull RtL"), (94, 86, "push RtL vs pull LtR"),
              (166, 167, "camera left vs right"), (168, 165, "camera up vs down"),
              (45, 43, "move up vs down"), (44, 41, "towards vs away camera"),
              (42, 40, "closer vs away from"), (37, 36, "closer vs apart (each other)"),
              (46, 5, "open vs close"), (6, 171, "cover vs uncover"), (14, 172, "fold vs unfold")],
}


def load_cosine():
    """Unit-normalise embeddings (as the wrapper does) and return the 174x174 cosine matrix + names."""
    emb = np.load(CFG["text_emb_path"]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    if CFG["center"]:                                   # remove the shared cone direction
        emb -= emb.mean(axis=0, keepdims=True)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = json.load(open(CFG["labels_path"]))
    names = [n for n, _ in sorted(labels.items(), key=lambda kv: int(kv[1]))]
    return emb @ emb.T, names


def pair_table(S, names):
    """Cosine, percentile among all off-diagonal pairs, and mutual neighbour ranks for each contrast pair."""
    off = S[~np.eye(len(S), dtype=bool)]
    order = np.argsort(-S, axis=1)                      # col 0 is self
    rank = lambda i, j: int(np.where(order[i] == j)[0][0])  # 1 = nearest neighbour
    rows = [dict(contrast=c, a=names[a], b=names[b], cosine=S[a, b],
                 pct_of_all=(off < S[a, b]).mean() * 100, rank_b_in_a=rank(a, b), rank_a_in_b=rank(b, a))
            for a, b, c in CFG["pairs"]]
    return pd.DataFrame(rows)


def verb_table(S, names):
    """Group classes by first word (verb). Per group: mean within- vs between-group cosine,
    and how often each member's nearest neighbour shares its verb (vs chance)."""
    verbs = np.array([n.split()[0].lower() for n in names])
    nn = np.argsort(-S, axis=1)[:, 1]
    rows = []
    for v in sorted(set(verbs)):
        m = verbs == v
        if m.sum() < 2:
            continue
        within = S[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)].mean()
        rows.append(dict(verb=v, n=int(m.sum()), within=within, between=S[np.ix_(m, ~m)].mean(),
                         nn_same_verb=(verbs[nn[m]] == v).mean(), nn_chance=(m.sum() - 1) / (len(S) - 1)))
    df = pd.DataFrame(rows)
    df["gap"] = df.within - df.between
    return df.sort_values("gap", ascending=False)


def main():
    S, names = load_cosine()
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"off-diagonal cosine: mean {off.mean():.3f}  median {np.median(off):.3f}  "
          f"p5 {np.percentile(off, 5):.3f}  p95 {np.percentile(off, 95):.3f}  max {off.max():.3f}")
    pairs, verbs = pair_table(S, names), verb_table(S, names)
    CFG["out_dir"].mkdir(parents=True, exist_ok=True)
    pairs.to_csv(CFG["out_dir"] / "contrast_pairs.csv", index=False)
    verbs.to_csv(CFG["out_dir"] / "verb_groups.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    print(pairs.drop(columns=["a", "b"]).round(3).to_string(index=False))
    print(verbs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

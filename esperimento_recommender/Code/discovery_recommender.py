"""Shared quantity, core-filtering, and deterministic ranking definitions."""
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
USER='Survey ResponseID'
ITEM='ASIN/ISBN (Product Code)'
DATE='Order Date'
QTY='Quantity'
CATEGORY='Category'
CATEGORY_SMOOTH_STRENGTH=20.0

def gini(values):
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0 or np.isclose(x.sum(), 0):
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((n + 1 - 2 * np.sum(np.cumsum(x)) / x.sum()) / n)

def head_tail_breaks(values, max_breaks=None, head_max_share=0.40):
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    current = x.copy()
    breaks, info = [], []
    level = 1

    while len(current) >= 2:
        b = float(current.mean())
        tail = current[current <= b]
        head = current[current > b]
        if len(tail) == 0 or len(head) == 0:
            break

        head_share = len(head)/len(current)
        breaks.append(b)
        info.append({
            "break_number": level,
            "break_value": b,
            "n_current": len(current),
            "n_lower": len(tail),
            "n_head": len(head),
            "head_share_within_current": head_share,
        })

        if max_breaks is not None and len(breaks) >= max_breaks:
            break
        if head_share > head_max_share:
            break
        current = head
        level += 1

    return breaks, pd.DataFrame(info)

def htb_three_class(values):
    s = pd.Series(values)
    breaks, info = head_tail_breaks(s, max_breaks=2)
    if len(breaks) < 2:
        raise ValueError(
            "HTB non ha prodotto due break: non introduco una soglia esterna "
            "per forzare artificialmente Head/Mid/Tail."
        )
    b1, b2 = breaks[:2]
    labels = pd.Series(
        np.select(
            [s <= b1, (s > b1) & (s <= b2), s > b2],
            ["tail", "mid", "head"],
            default="unknown"
        ),
        index=s.index, dtype="object"
    )
    return labels, [b1, b2], info

def iterative_kcore_pairs(pairs, k=5):
    x = pairs[[USER, ITEM]].drop_duplicates().copy()
    while True:
        old = (len(x), x[USER].nunique(), x[ITEM].nunique())
        uc = x.groupby(USER)[ITEM].nunique()
        x = x[x[USER].isin(uc[uc >= k].index)]
        ic = x.groupby(ITEM)[USER].nunique()
        x = x[x[ITEM].isin(ic[ic >= k].index)]
        new = (len(x), x[USER].nunique(), x[ITEM].nunique())
        if new == old:
            break
    return x

class TopPopularQuantity:
    def __init__(self, q):
        self.ranked = list(q.sort_values(ascending=False).index)

    def recommend(self, user_id, seen, k=10, **kwargs):
        seen = seen or set()
        return [i for i in self.ranked if i not in seen][:k]

class ItemKNNFast:
    """
    ItemKNN cosine con vicini calcolati ON DEMAND e messi in cache.

    Evita `NearestNeighbors(...).kneighbors(all_items)`, che materializzava una
    ricerca brute-force per ogni item del catalogo. Il vicinato di un item viene
    calcolato soltanto quando quell'item compare nella storia di un buyer valutato.
    La top-N similarity è la stessa cosine similarity binaria prevista da ItemKNN.
    """
    def __init__(self, pairs, fallback_quantity, n_neighbors=100):
        pairs = pairs[[USER, ITEM]].drop_duplicates()
        self.users = sorted(pairs[USER].unique())
        self.items = sorted(pairs[ITEM].unique())
        self.u2i = {u: j for j, u in enumerate(self.users)}
        self.i2i = {it: j for j, it in enumerate(self.items)}
        self.idx2item = {j: it for it, j in self.i2i.items()}

        rr = pairs[USER].map(self.u2i).to_numpy()
        cc = pairs[ITEM].map(self.i2i).to_numpy()
        X = sparse.csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (rr, cc)),
            shape=(len(self.users), len(self.items)),
        )
        X.data[:] = 1.0
        self.X = X
        self.item_user = normalize(X.T.tocsr(), norm="l2", axis=1, copy=True).astype(np.float32)
        self.max_neighbors = min(int(n_neighbors), max(len(self.items) - 1, 1))
        item_set = set(self.items)
        self.fallback = [i for i in fallback_quantity.sort_values(ascending=False).index if i in item_set]
        self._neighbor_cache = {}

    def _neighbors(self, item_idx):
        cached = self._neighbor_cache.get(item_idx)
        if cached is not None:
            return cached

        # Similarità del solo item richiesto contro il catalogo.
        q = self.item_user.getrow(item_idx).T
        sim = (self.item_user @ q).tocoo()
        inds = sim.row.astype(np.int32, copy=False)
        vals = sim.data.astype(np.float32, copy=False)
        mask = (inds != item_idx) & (vals > 0)
        inds, vals = inds[mask], vals[mask]

        if len(vals) > self.max_neighbors:
            keep = np.argpartition(vals, -self.max_neighbors)[-self.max_neighbors:]
            inds, vals = inds[keep], vals[keep]
        if len(vals):
            order = np.argsort(-vals)
            inds, vals = inds[order], vals[order]

        self._neighbor_cache[item_idx] = (inds, vals)
        return inds, vals

    def recommend(self, user_id, seen, k=10, neighbor_limit=None):
        seen = seen or set()
        limit = self.max_neighbors if neighbor_limit is None else min(int(neighbor_limit), self.max_neighbors)
        scores = defaultdict(float)
        for it in seen:
            ii = self.i2i.get(it)
            if ii is None:
                continue
            inds, vals = self._neighbors(ii)
            if limit < len(inds):
                inds, vals = inds[:limit], vals[:limit]
            for j, sim in zip(inds, vals):
                cand = self.idx2item[int(j)]
                if cand not in seen:
                    scores[cand] += float(sim)

        if scores:
            ranked = [i for i, _ in sorted(scores.items(), key=lambda z: (-z[1], str(z[0])))]
        else:
            ranked = []
        if len(ranked) < k:
            used = set(ranked) | set(seen)
            ranked.extend([i for i in self.fallback if i not in used][:k-len(ranked)])
        return ranked[:k]

class CategoryConditionedQuantity:
    """Versione vettorizzata della category-conditioned popularity."""
    def __init__(self, ctx, smooth_strength=CATEGORY_SMOOTH_STRENGTH):
        self.catalog = list(ctx["core_items"])
        self.item_to_idx = {it: j for j, it in enumerate(self.catalog)}
        self.q = ctx["train_quantity_precore"]
        # Determine product categories from provisional training only.
        train_cat = ctx["provisional_train"].groupby([ITEM, "category"]).size().rename("n").reset_index()
        primary = (train_cat.sort_values([ITEM, "n", "category"], ascending=[True, False, True])
                   .drop_duplicates(ITEM).set_index(ITEM)["category"].to_dict())
        self.item_cat = {i: primary.get(i, "UNKNOWN") for i in self.catalog}

        tr = ctx["provisional_train"]
        tr = tr[tr[USER].isin(ctx["core_users"])].copy()
        tr["cat"] = tr["category"].astype("string")
        gc = tr.groupby("cat")["quantity"].sum()
        global_cat = gc / gc.sum()
        self.user_cat = tr.groupby([USER, "cat"])["quantity"].sum()
        self.user_total = self.user_cat.groupby(level=0).sum()
        self.smooth = float(smooth_strength)

        cats = sorted(set(self.item_cat.values()) | set(global_cat.index.astype(str)))
        self.cat_to_idx = {c: j for j, c in enumerate(cats)}
        self.global_cat_arr = np.array([float(global_cat.get(c, 0.0)) for c in cats], dtype=np.float64)
        self.item_cat_idx = np.array([self.cat_to_idx[self.item_cat[i]] for i in self.catalog], dtype=np.int32)

        ci = pd.DataFrame({
            ITEM: self.catalog,
            "cat": [self.item_cat[i] for i in self.catalog],
            "q": [self.q.get(i, 0.0) for i in self.catalog],
        })
        denom = ci.groupby("cat")["q"].transform("sum").replace(0, np.nan)
        self.within_arr = (ci["q"] / denom).fillna(0).to_numpy(dtype=np.float64)

    def recommend(self, user_id, seen, k=10, **kwargs):
        n = float(self.user_total.get(user_id, 0.0))
        pref = self.smooth * self.global_cat_arr.copy()
        try:
            uc = self.user_cat.loc[user_id]
            for cat, value in uc.items():
                j = self.cat_to_idx.get(str(cat))
                if j is not None:
                    pref[j] += float(value)
        except KeyError:
            pass
        pref /= (n + self.smooth)
        score = pref[self.item_cat_idx] * self.within_arr
        score = score.copy()
        for it in seen or set():
            j = self.item_to_idx.get(it)
            if j is not None:
                score[j] = -np.inf
        kk = min(int(k), len(score))
        if kk <= 0:
            return []
        top = np.argpartition(-score, kk-1)[:kk]
        top = top[np.argsort(-score[top])]
        return [self.catalog[int(j)] for j in top]


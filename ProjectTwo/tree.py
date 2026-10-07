import numpy as np


# --------------------------------------------------------------------------
# Tree structure
# --------------------------------------------------------------------------
class DecisionTree:
    EPS = 1e-12

    def __init__(self, task, is_numeric, n_classes=None, feature_names=None):
        assert task in ("classification", "regression")
        self.task = task
        self.is_numeric = list(is_numeric)
        self.K = n_classes
        self.names = feature_names or [f"f{i}" for i in range(len(is_numeric))]
        self.root = None

    # ------------------------------ training ------------------------------
    def fit(self, X, y, idx=None):
        """X: list of column arrays (float for numeric, object/str for categorical)."""
        idx = np.arange(len(y)) if idx is None else np.asarray(idx)
        avail = frozenset(range(len(self.is_numeric)))
        self.root = self._build(X, y, idx, avail)
        return self

    def _leaf_value(self, yi):
        if self.task == "classification":
            return int(np.argmax(np.bincount(yi, minlength=self.K)))
        return float(yi.mean())

    def _build(self, X, y, idx, avail):
        yi = y[idx]
        node = Node(self._leaf_value(yi), len(idx))
        if len(idx) < 2 or not avail:
            return node
        if self.task == "classification" and np.all(yi == yi[0]):
            return node
        if self.task == "regression" and yi.var() < self.EPS:
            return node

        best = self._best_split(X, y, idx, avail)
        if best is None:
            return node
        feat, thr = best
        node.feat, node.thr = feat, thr
        node.numeric = self.is_numeric[feat]
        col = X[feat][idx]
        if node.numeric:
            m = col <= thr
            node.children = [self._build(X, y, idx[m], avail),
                             self._build(X, y, idx[~m], avail)]
        else:
            rest = avail - {feat}
            node.children = {v: self._build(X, y, idx[col == v], rest)
                             for v in np.unique(col)}
        return node

    def _best_split(self, X, y, idx, avail):
        best_score, best = -np.inf, None
        yi = y[idx]
        n = len(idx)
        if self.task == "regression":
            node_mse = yi.var()
        for f in avail:
            col = X[f][idx]
            if self.is_numeric[f]:
                res = (self._clf_numeric(col, yi) if self.task == "classification"
                       else self._reg_numeric(col, yi))
            else:
                res = (self._clf_categorical(col, yi) if self.task == "classification"
                       else self._reg_categorical(col, yi))
            if res is None:
                continue
            score, thr = res  # higher = better (we negate MSE for regression)
            if score > best_score:
                best_score, best = score, (f, thr)
        if best is None:
            return None
        if self.task == "regression" and -best_score >= node_mse - self.EPS:
            return None  # no improvement in MSE
        return best

    # ---- classification: gain ratio ----
    def _clf_numeric(self, col, yi):
        order = np.argsort(col, kind="stable")
        v, ys = col[order], yi[order]
        n = len(v)
        pos = np.nonzero(v[:-1] < v[1:])[0]
        if len(pos) == 0:
            return None
        cum = np.cumsum(np.eye(self.K)[ys], axis=0)
        tot = cum[-1]
        L = cum[pos]
        R = tot - L
        nl = (pos + 1).astype(float)
        nr = n - nl
        H = entropyRows(tot[None, :])[0]
        gain = H - (nl / n) * entropyRows(L) - (nr / n) * entropyRows(R)
        wl, wr = nl / n, nr / n
        iv = -(wl * np.log2(wl) + wr * np.log2(wr))
        ok = (gain > self.EPS) & (iv > self.EPS)
        if not ok.any():
            return None
        ratio = np.where(ok, gain / np.maximum(iv, self.EPS), -np.inf)
        k = int(np.argmax(ratio))
        p = pos[k]
        thr = (v[p] + v[p + 1]) / 2.0
        if thr >= v[p + 1]:
            thr = v[p]
        return ratio[k], thr

    def _clf_categorical(self, col, yi):
        vals, inv = np.unique(col, return_inverse=True)
        if len(vals) < 2:
            return None
        n = len(yi)
        C = np.zeros((len(vals), self.K))
        np.add.at(C, (inv, yi), 1)
        nj = C.sum(axis=1)
        H = entropyRows(C.sum(axis=0)[None, :])[0]
        gain = H - ((nj / n) * entropyRows(C)).sum()
        w = nj / n
        iv = -(w * np.log2(w)).sum()
        if gain <= self.EPS or iv <= self.EPS:
            return None
        return gain / iv, None

    # ---- regression: weighted MSE (returned negated) ----
    def _reg_numeric(self, col, yi):
        order = np.argsort(col, kind="stable")
        v, ys = col[order], yi[order]
        n = len(v)
        pos = np.nonzero(v[:-1] < v[1:])[0]
        if len(pos) == 0:
            return None
        cs, cq = np.cumsum(ys), np.cumsum(ys ** 2)
        nl = (pos + 1).astype(float)
        nr = n - nl
        sl, ql = cs[pos], cq[pos]
        sr, qr = cs[-1] - sl, cq[-1] - ql
        sse = (ql - sl ** 2 / nl) + (qr - sr ** 2 / nr)
        k = int(np.argmin(sse))
        p = pos[k]
        thr = (v[p] + v[p + 1]) / 2.0
        if thr >= v[p + 1]:
            thr = v[p]
        return -sse[k] / n, thr

    def _reg_categorical(self, col, yi):
        vals, inv = np.unique(col, return_inverse=True)
        if len(vals) < 2:
            return None
        cnt = np.bincount(inv).astype(float)
        s = np.bincount(inv, weights=yi)
        q = np.bincount(inv, weights=yi ** 2)
        sse = (q - s ** 2 / cnt).sum()
        return -sse / len(yi), None

    # ----------------------------- prediction -----------------------------
    def _partition(self, node, X, idx):
        """Split row indices among node's children -> list of (child|None, idx)."""
        col = X[node.feat][idx]
        if node.numeric:
            m = col <= node.thr
            return [(node.children[0], idx[m]), (node.children[1], idx[~m])]
        parts, assigned = [], np.zeros(len(idx), dtype=bool)
        for v, ch in node.children.items():
            m = col == v
            parts.append((ch, idx[m]))
            assigned |= m
        if (~assigned).any():  # unseen category -> stop at this node
            parts.append((None, idx[~assigned]))
        return parts

    def predict(self, X, n):
        out = np.empty(n, dtype=float)
        self._predict(self.root, X, np.arange(n), out)
        return out

    def _predict(self, node, X, idx, out):
        if len(idx) == 0:
            return
        if node.is_leaf:
            out[idx] = node.predictions
            return
        for ch, sub in self._partition(node, X, idx):
            if ch is None:
                out[sub] = node.predictions
            else:
                self._predict(ch, X, sub, out)

    # ------------------------------ pruning -------------------------------
    def _err(self, yv, pred):
        if self.task == "classification":
            return float((yv != pred).sum())
        return float(((yv - pred) ** 2).sum())

    def prune(self, X, y):
        """Reduced-error pruning on the pruning set (X, y)."""
        self._prune(self.root, X, y, np.arange(len(y)))
        return self

    def _prune(self, node, X, y, idx):
        leaf_err = self._err(y[idx], node.predictions)
        if node.is_leaf:
            return leaf_err
        sub_err = 0.0
        for ch, sub in self._partition(node, X, idx):
            if ch is None:
                sub_err += self._err(y[sub], node.predictions)
            else:
                sub_err += self._prune(ch, X, y, sub)
        if leaf_err <= sub_err:  # leaf is no worse -> collapse
            node.feat = node.thr = node.children = None
            node.numeric = False
            return leaf_err
        return sub_err

    # ------------------------------ utilities -----------------------------
    def n_nodes(self, node=None):
        node = node or self.root
        return 1 + sum(self.n_nodes(c) for c in node.child_nodes())

    def print_tree(self, class_names=None, node=None, depth=0, max_depth=6):
        node = node or self.root
        pad = "  " * depth
        pred = (class_names[node.predictions] if class_names is not None and
                                          self.task == "classification" else f"{node.predictions:.3g}")
        if node.is_leaf or depth >= max_depth:
            print(f"{pad}-> {pred}  (n={node.n})")
            return
        nm = self.names[node.feat]
        if node.numeric:
            print(f"{pad}{nm} <= {node.thr:.4g}:")
            self.print_tree(class_names, node.children[0], depth + 1, max_depth)
            print(f"{pad}{nm} >  {node.thr:.4g}:")
            self.print_tree(class_names, node.children[1], depth + 1, max_depth)
        else:
            for v, ch in node.children.items():
                print(f"{pad}{nm} == {v}:")
                self.print_tree(class_names, ch, depth + 1, max_depth)






class Node:
    def __init__(self, predictions,n):
        self.predictions = predictions
        self.n = n
        self.feat = None
        self.numeric = False
        self.thr = None
        self.children = None

    @property
    def is_leaf(self):
        return self.feat is None

    def child_nodes(self):
        if self.is_leaf:
            return []
        elif self.numeric:
            return self.children
        return list(self.children.values())


def entropyRows(C):
    n = C.sum(axis=1, keepdims=True)
    p = np.where(C > 0, C / np.maximum(n, 1), 1.0)
    return -(np.where(C > 0, p * np.log2(p), 0.0)).sum(axis=1)
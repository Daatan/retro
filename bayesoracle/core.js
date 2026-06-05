/*
 * BayesOracle core engine — browser port of core.py.
 *
 * Faithful port of the Python engine so the HTML viewers, the API and the
 * backtest all run the *same* inference: fitted-intercept logistic CPT with
 * exact enumeration over parent states, plus exclusive-group simplex
 * normalisation.  Numerical parity with core.py is checked in CI-style by
 * verify_parity (see bayesoracle/verify_core_parity.py).
 *
 * Usage:
 *   const g = new BayesGraph(spec);   // spec = {nodes, edges, exclusive_groups}
 *   const p = g.propagate({NODE: 0.9});  // -> {id: probability}
 *
 * Exposes a global `BayesGraph` (and `Bayes` namespace) when loaded via a
 * <script> tag.
 */
(function (root) {
  "use strict";

  var PROB_FLOOR = 0.001, PROB_CEIL = 0.999, EPS = 1e-9;

  function logit(p) {
    p = Math.min(PROB_CEIL, Math.max(PROB_FLOOR, p));
    return Math.log(p / (1 - p));
  }
  function sigmoid(x) {
    if (x >= 0) { var z = Math.exp(-x); return 1 / (1 + z); }
    var e = Math.exp(x); return e / (1 + e);
  }

  function BayesGraph(spec) {
    this.name = spec.name || "bayesoracle";
    this.nodes = spec.nodes;
    this.edges = spec.edges || [];
    this.exclusiveGroups = spec.exclusive_groups || [];

    this.prior = {};
    this.nodeById = {};
    var i;
    for (i = 0; i < this.nodes.length; i++) {
      var n = this.nodes[i];
      this.prior[n.id] = +n.prior;
      this.nodeById[n.id] = n;
    }

    // parents[target] -> [{source, weight, pYes, pNo, type}]
    this.parents = {};
    for (i = 0; i < this.edges.length; i++) {
      var e = this.edges[i];
      var w = logit(+e.pYes) - logit(+e.pNo);
      if (e.type === "secondary") w *= 0.5;
      (this.parents[e.target] || (this.parents[e.target] = [])).push({
        source: e.source, weight: w, pYes: +e.pYes, pNo: +e.pNo,
        type: e.type || "primary"
      });
    }

    this.validate();
    this.topo = this._topoOrder();

    this.intercept = {};
    for (i = 0; i < this.topo.length; i++) {
      this.intercept[this.topo[i]] = this._fitIntercept(this.topo[i]);
    }

    // group whose last (in topo) member triggers inline normalisation
    var pos = {};
    for (i = 0; i < this.topo.length; i++) pos[this.topo[i]] = i;
    this.groupAfter = {};
    for (i = 0; i < this.exclusiveGroups.length; i++) {
      var grp = this.exclusiveGroups[i];
      var members = grp.members.filter(function (m) { return m in pos; });
      if (!members.length) continue;
      var last = members.reduce(function (a, b) { return pos[a] >= pos[b] ? a : b; });
      (this.groupAfter[last] || (this.groupAfter[last] = [])).push(grp);
    }
  }

  BayesGraph.prototype.validate = function () {
    var ids = this.nodeById, i;
    for (i = 0; i < this.nodes.length; i++) {
      var p = this.nodes[i].prior;
      if (typeof p !== "number" || p < 0 || p > 1)
        throw new Error("node " + this.nodes[i].id + " prior out of [0,1]: " + p);
    }
    for (i = 0; i < this.edges.length; i++) {
      var e = this.edges[i];
      if (!(e.source in ids)) throw new Error("edge source not a node: " + e.source);
      if (!(e.target in ids)) throw new Error("edge target not a node: " + e.target);
    }
    for (var tgt in this.parents)
      if (this.parents[tgt].length > 16)
        throw new Error("fan-in > 16 at " + tgt);
  };

  BayesGraph.prototype._enumerate = function (parents, b, pSource) {
    if (!parents || !parents.length) return sigmoid(b);
    var k = parents.length, total = 0, mask;
    for (mask = 0; mask < (1 << k); mask++) {
      var pCombo = 1, z = b, j;
      for (j = 0; j < k; j++) {
        var on = (mask >> j) & 1;
        var pi = pSource[parents[j].source];
        pCombo *= on ? pi : (1 - pi);
        if (on) z += parents[j].weight;
      }
      if (pCombo > EPS) total += pCombo * sigmoid(z);
    }
    return total;
  };

  BayesGraph.prototype._fitIntercept = function (id) {
    var ps = this.parents[id];
    if (!ps || !ps.length) return 0;
    var target = Math.min(PROB_CEIL, Math.max(PROB_FLOOR, this.prior[id]));
    var pSource = {}, i;
    for (i = 0; i < ps.length; i++) pSource[ps[i].source] = this.prior[ps[i].source];
    var lo = -40, hi = 40;
    for (i = 0; i < 100; i++) {
      var mid = (lo + hi) / 2;
      if (this._enumerate(ps, mid, pSource) < target) lo = mid; else hi = mid;
      if (hi - lo < 1e-10) break;
    }
    return (lo + hi) / 2;
  };

  BayesGraph.prototype._topoOrder = function () {
    var indeg = {}, children = {}, nid;
    for (nid in this.nodeById) { indeg[nid] = 0; children[nid] = []; }
    for (var tgt in this.parents) {
      var ps = this.parents[tgt];
      for (var i = 0; i < ps.length; i++) {
        indeg[tgt]++;
        children[ps[i].source].push(tgt);
      }
    }
    var self = this;
    var key = function (a) {
      var la = (self.nodeById[a].layer || 0), lb;
      return la;
    };
    var ready = [];
    for (nid in indeg) if (indeg[nid] === 0) ready.push(nid);
    var sortReady = function () {
      ready.sort(function (a, b) {
        var la = self.nodeById[a].layer || 0, lb = self.nodeById[b].layer || 0;
        return la !== lb ? la - lb : (a < b ? -1 : a > b ? 1 : 0);
      });
    };
    sortReady();
    var order = [];
    while (ready.length) {
      var x = ready.shift();
      order.push(x);
      var cs = children[x];
      for (var c = 0; c < cs.length; c++) {
        if (--indeg[cs[c]] === 0) ready.push(cs[c]);
      }
      sortReady();
    }
    if (order.length !== Object.keys(indeg).length)
      throw new Error("graph has a cycle");
    return order;
  };

  BayesGraph.prototype.propagate = function (observations) {
    var current = {}, locked = {}, nid;
    for (nid in this.prior) current[nid] = this.prior[nid];
    if (observations) {
      for (nid in observations) {
        if (nid in current) {
          current[nid] = Math.min(1, Math.max(0, +observations[nid]));
          locked[nid] = true;
        }
      }
    }
    for (var i = 0; i < this.topo.length; i++) {
      nid = this.topo[i];
      if (!(nid in locked)) {
        var ps = this.parents[nid];
        if (ps && ps.length) current[nid] = this._enumerate(ps, this.intercept[nid], current);
      }
      var grps = this.groupAfter[nid];
      if (grps) for (var g = 0; g < grps.length; g++) normaliseGroup(current, grps[g], locked);
    }
    return current;
  };

  // Predict each node purely from its parents (linear LToTP; primary weight 1,
  // secondary 0.5) without anchoring to the node's own prior — the divergence
  // model used by pm_analysis. Mirrors core.py reconcile().
  BayesGraph.prototype.reconcile = function (values) {
    var current = {}, nid;
    for (nid in this.prior) current[nid] = this.prior[nid];
    if (values) for (nid in values)
      if (nid in current) current[nid] = Math.min(1, Math.max(0, +values[nid]));
    for (var i = 0; i < this.topo.length; i++) {
      nid = this.topo[i];
      var ps = this.parents[nid];
      if (ps && ps.length) {
        var wsum = 0, tw = 0;
        for (var j = 0; j < ps.length; j++) {
          var w = ps[j].type === "secondary" ? 0.5 : 1.0;
          var psrc = current[ps[j].source];
          wsum += w * (ps[j].pYes * psrc + ps[j].pNo * (1 - psrc));
          tw += w;
        }
        if (tw) current[nid] = wsum / tw;
      }
      var grps = this.groupAfter[nid];
      if (grps) for (var k = 0; k < grps.length; k++) normaliseGroup(current, grps[k], {});
    }
    return current;
  };

  function normaliseGroup(current, grp, locked) {
    var members = grp.members.filter(function (m) { return m in current; });
    var free = members.filter(function (m) { return !(m in locked); });
    var lockedMass = 0, i;
    for (i = 0; i < members.length; i++) if (members[i] in locked) lockedMass += current[members[i]];
    var budget = Math.max(0, 1 - lockedMass);
    var s = 0;
    for (i = 0; i < free.length; i++) s += current[free[i]];
    if (s <= EPS) return;
    var allowOther = grp.allow_other !== false;
    if (allowOther) {
      if (s > budget) for (i = 0; i < free.length; i++) current[free[i]] *= budget / s;
    } else {
      for (i = 0; i < free.length; i++) current[free[i]] *= budget / s;
    }
  }

  var Bayes = { BayesGraph: BayesGraph, logit: logit, sigmoid: sigmoid };
  root.BayesGraph = BayesGraph;
  root.Bayes = Bayes;
  if (typeof module !== "undefined" && module.exports) module.exports = Bayes;
})(typeof window !== "undefined" ? window : this);

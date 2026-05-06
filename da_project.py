import argparse, math, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def deferred_acceptance(doc_prefs, hosp_prefs):
    """Doctor-proposing Gale-Shapley DA. Returns matching and proposal count."""
    n = len(doc_prefs)
    hosp_rank = np.empty((n, n), dtype=int)
    for h in range(n):
        for r, d in enumerate(hosp_prefs[h]):
            hosp_rank[h, d] = r

    free = list(range(n))
    next_choice = [0] * n
    match_h = [-1] * n      # hospital -> doctor
    match_d = [-1] * n      # doctor -> hospital
    proposals = 0

    while free:
        d = free.pop()
        h = doc_prefs[d][next_choice[d]]
        next_choice[d] += 1
        proposals += 1

        old = match_h[h]
        if old == -1:
            match_h[h] = d
            match_d[d] = h
        elif hosp_rank[h, d] < hosp_rank[h, old]:
            match_h[h] = d
            match_d[d] = h
            match_d[old] = -1
            free.append(old)
        else:
            free.append(d)

    return match_d, match_h, proposals


def random_prefs(n, rng):
    return [rng.permutation(n).tolist() for _ in range(n)]


def public_private_prefs(n, lam, rng):
    public = rng.random(n)
    prefs = []
    for _ in range(n):
        private = rng.random(n)
        util = lam * public + (1.0 - lam) * private
        prefs.append(np.argsort(-util).tolist())
    return prefs


def avg_partner_ranks(match_d, match_h, doc_prefs, hosp_prefs):
    n = len(doc_prefs)
    doc_ranks = []
    hosp_ranks = []
    for d, h in enumerate(match_d):
        doc_ranks.append(doc_prefs[d].index(h) + 1)
    for h, d in enumerate(match_h):
        hosp_ranks.append(hosp_prefs[h].index(d) + 1)
    return float(np.mean(doc_ranks)), float(np.mean(hosp_ranks))


def run_uniform(n_values, runs, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        prop, dr, hr = [], [], []
        for _ in range(runs):
            doc_prefs = random_prefs(n, rng)
            hosp_prefs = random_prefs(n, rng)
            match_d, match_h, p = deferred_acceptance(doc_prefs, hosp_prefs)
            d_rank, h_rank = avg_partner_ranks(match_d, match_h, doc_prefs, hosp_prefs)
            prop.append(p); dr.append(d_rank); hr.append(h_rank)
        rows.append((n, np.mean(prop), np.mean(dr), np.mean(hr), math.log(n), n / math.log(n)))
    return rows


def run_public_private(n, lambdas, runs, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for lam in lambdas:
        prop, dr, hr = [], [], []
        for _ in range(runs):
            doc_prefs = public_private_prefs(n, lam, rng)
            hosp_prefs = public_private_prefs(n, lam, rng)
            match_d, match_h, p = deferred_acceptance(doc_prefs, hosp_prefs)
            d_rank, h_rank = avg_partner_ranks(match_d, match_h, doc_prefs, hosp_prefs)
            prop.append(p); dr.append(d_rank); hr.append(h_rank)
        rows.append((lam, np.mean(prop), np.mean(dr), np.mean(hr)))
    return rows


def save_csv(path, header, rows):
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')


def make_plots(out, uniform_rows, pp_rows):
    n = [r[0] for r in uniform_rows]
    prop = [r[1] for r in uniform_rows]
    doc = [r[2] for r in uniform_rows]
    hosp = [r[3] for r in uniform_rows]
    logn = [r[4] for r in uniform_rows]
    nlogn = [r[5] for r in uniform_rows]

    plt.figure()
    plt.plot(n, prop, marker='o')
    plt.xlabel('n')
    plt.ylabel('average total proposals')
    plt.title('Doctor-proposing DA: average proposals')
    plt.tight_layout()
    plt.savefig(out / 'uniform_proposals.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(n, doc, marker='o', label='doctor average rank')
    plt.plot(n, logn, marker='o', label='log n')
    plt.xlabel('n')
    plt.ylabel('rank')
    plt.title('Doctor ranks under uniform random preferences')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'uniform_doctor_rank.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(n, hosp, marker='o', label='hospital average rank')
    plt.plot(n, nlogn, marker='o', label='n / log n')
    plt.xlabel('n')
    plt.ylabel('rank')
    plt.title('Hospital ranks under uniform random preferences')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'uniform_hospital_rank.png', dpi=200)
    plt.close()

    lam = [r[0] for r in pp_rows]
    pp_prop = [r[1] for r in pp_rows]
    pp_doc = [r[2] for r in pp_rows]
    pp_hosp = [r[3] for r in pp_rows]

    plt.figure()
    plt.plot(lam, pp_prop, marker='o')
    plt.xlabel('lambda')
    plt.ylabel('average total proposals')
    plt.title('Public-private model: proposals')
    plt.tight_layout()
    plt.savefig(out / 'pp_proposals.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(lam, pp_doc, marker='o', label='doctor average rank')
    plt.plot(lam, pp_hosp, marker='o', label='hospital average rank')
    plt.xlabel('lambda')
    plt.ylabel('average partner rank')
    plt.title('Public-private model: average partner ranks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / 'pp_ranks.png', dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--n-values', type=int, nargs='+', default=[20, 50, 100, 200, 400])
    parser.add_argument('--pp-n', type=int, default=200)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    uniform = run_uniform(args.n_values, args.runs, args.seed)
    pp = run_public_private(args.pp_n, args.lambdas, args.runs, args.seed + 100)

    save_csv(out / 'uniform_results.csv', ['n', 'avg_proposals', 'doctor_avg_rank', 'hospital_avg_rank', 'log_n', 'n_over_log_n'], uniform)
    save_csv(out / 'public_private_results.csv', ['lambda', 'avg_proposals', 'doctor_avg_rank', 'hospital_avg_rank'], pp)
    make_plots(out, uniform, pp)

    print('Uniform results')
    for r in uniform:
        print(r)
    print('\nPublic-private results')
    for r in pp:
        print(r)
    print(f'\nSaved results to {out.resolve()}')


if __name__ == '__main__':
    main()

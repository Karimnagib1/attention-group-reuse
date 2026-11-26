"""Validate the completed sweep and extend the paper's similarity statistics.

Run from any directory with Python, NumPy, SciPy, and PyYAML. Reads saved model
outputs only; writes summary.json and language_pairs.csv beside this script.
Uses the paper's existing bootstrap and permutation implementations, with an
independent SciPy check of aligned Spearman and a set-based Jaccard check.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "ministral_outputs/Ministral-3-14B-Base-2512/belebele"
spec = importlib.util.spec_from_file_location(
    "paper_statistics", ROOT / "scripts/statistical_support/compute_paper_statistics.py"
)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def main():
    samples = json.loads((RUN / "samples.json").read_text())
    metadata = json.loads((RUN / "metadata.json").read_text())
    saved_spearman = json.loads((RUN / "spearman_summary.json").read_text())
    saved_jaccard = json.loads((RUN / "jaccard_summary.json").read_text())
    archive = np.load(RUN / "importance_tensors.npz")
    assert all(np.isfinite(archive[key]).all() for key in archive.files)
    assert len({(s['parallel_id'], s['language']) for s in samples}) == len(samples)
    assert all(s['n_choice_order_variants'] == 4 for s in samples)
    predictions = [json.loads(line) for line in (RUN / "predictions.jsonl").read_text().splitlines()]
    assert len({(r['id'], r['language']) for r in predictions}) == len(predictions) == 4500
    by_id = defaultdict(list)
    correct = Counter()
    for row in predictions:
        by_id[row['id']].append(row)
        correct[row['language']] += int(row['correct'])
    eligible = {pid for pid, rows in by_id.items() if len(rows) == 5 and all(r['correct'] for r in rows)}
    assert eligible == set(json.loads((RUN / 'all_correct_ids.json').read_text()))
    assert eligible == {s['parallel_id'] for s in samples}
    assert sum(correct.values()) == json.loads((RUN / 'metrics.json').read_text())['correct']

    metric = "pct_var_mean_incorrect"
    tensor, languages, pids = stats.load_balanced_tensor(RUN, metric)
    normalized, valid = stats.rank_normalized_rows(tensor)
    assert valid.all()
    binary = stats.jaccard_binary_rows(tensor, 10)
    independent = rankdata(tensor, axis=2, method="average")
    independent -= independent.mean(axis=2, keepdims=True)
    independent /= np.linalg.norm(independent, axis=2, keepdims=True)
    counts = stats.make_bootstrap_counts(len(pids), 10000, stats.stable_seed(42, "Ministral-3-14B-Base-2512", "Belebele"))
    matrices = {key: [] for key in ['spearman', 'jaccard@10']}
    rows = []
    max_spearman_error = 0.0
    for first in range(len(languages)):
        for second in range(first + 1, len(languages)):
            pair = (languages[first], languages[second])
            pair_matrices = {
                'spearman': stats.spearman_similarity_matrix(normalized, valid, first, second),
                'jaccard@10': stats.jaccard_similarity_matrix(binary, first, second, 10),
            }
            independent_rho = np.sum(independent[first] * independent[second], axis=1)
            expected = saved_spearman[metric]['mean_spearman'][first][second]
            error = abs(float(independent_rho.mean()) - expected)
            max_spearman_error = max(max_spearman_error, error)
            assert error < 1e-6
            np.testing.assert_allclose(np.diag(pair_matrices['spearman']), independent_rho, atol=1e-6)
            sets_a = [set(np.flatnonzero(row)) for row in binary[first]]
            sets_b = [set(np.flatnonzero(row)) for row in binary[second]]
            independent_j = [len(a & b) / len(a | b) for a, b in zip(sets_a, sets_b)]
            np.testing.assert_allclose(np.diag(pair_matrices['jaccard@10']), independent_j, atol=1e-7)
            # Also reconcile the sweep's top-5 result; there are no boundary ties at k=5.
            top5a = [set(np.argpartition(-row, 4)[:5]) for row in tensor[first]]
            top5b = [set(np.argpartition(-row, 4)[:5]) for row in tensor[second]]
            j5 = np.mean([len(a & b) / len(a | b) for a, b in zip(top5a, top5b)])
            assert abs(j5 - saved_jaccard[metric]['mean_jaccard'][first][second]) < 2e-6
            for name, matrix in pair_matrices.items():
                matrices[name].append(matrix)
                result = stats.bootstrap_similarity_matrix(matrix, counts, .95, 256)
                perm = stats.permutation_test(matrix, 10000, stats.stable_seed(123, 'Ministral-3-14B-Base-2512', 'Belebele', name, *pair), 256)
                rows.append({'language_a': pair[0], 'language_b': pair[1], 'similarity': name, **result, **perm})
            print('Completed', *pair, flush=True)
    adjusted = stats.holm_adjust([r['permutation_p_value'] for r in rows])
    for row, value in zip(rows, adjusted):
        row['permutation_p_value_holm_20_tests'] = value
    aggregate = {}
    for name, items in matrices.items():
        matrix = np.mean(items, axis=0)
        aggregate[name] = stats.bootstrap_similarity_matrix(matrix, counts, .95, 256)
        within = []
        for lang in range(len(languages)):
            same = (stats.spearman_similarity_matrix(normalized, valid, lang, lang)
                    if name == 'spearman' else stats.jaccard_similarity_matrix(binary, lang, lang, 10))
            within.append(float(same[~np.eye(len(pids), dtype=bool)].mean()))
        aggregate[name]['within_language_different_question_exact_mean'] = float(np.mean(within))
    rank = archive['flat_rank_drop_correct']
    sorted_values = np.sort(tensor, axis=2)
    payload = {
        'source': str(RUN.relative_to(ROOT)), 'job_id': 264947,
        'metadata': metadata,
        'baseline_accuracy': {'overall': sum(correct.values()) / len(predictions),
                              'by_language': {lang: correct[lang] / 900 for lang in languages}},
        'settings': {'importance_metric': metric, 'jaccard_k': 10,
                     'bootstrap_resamples': 10000, 'permutation_draws': 10000,
                     'bootstrap_seed': 42, 'permutation_seed': 123, 'ci_level': .95,
                     'holm_family': '20 tests: this model/dataset, ten pairs, two metrics',
                     'cross_language_control': 'Exact expected mean under unrestricted permutations, including fixed points; matches paper statistical-support implementation.',
                     'within_language_control': 'Exact mean over different questions, excluding fixed points.',
                     'bootstrap_unit': 'Question, preserving all five languages; recomputes aligned and expected control statistics.'},
        'validation': {'all_tensors_finite': True, 'unique_balanced_samples': True,
                       'evaluation_and_eligible_ids_reconciled': True,
                       'saved_spearman_max_absolute_error': max_spearman_error,
                       'top10_boundary_tie_samples': int(np.sum(sorted_values[:, :, -10] == sorted_values[:, :, -11]))},
        'aggregate': aggregate, 'language_pairs': rows,
        'rank_drop_caveat': {'zero_entry_fraction': float(np.mean(rank == 0)),
                             'all_zero_vectors': int(np.all(rank == 0, axis=1).sum()),
                             'total_vectors': len(rank),
                             'saved_top5_jaccard': saved_jaccard['rank_drop_correct']['overall_mean_jaccard_across_language_pairs'],
                             'interpretation': 'Tied all-zero vectors yield arbitrary identical top-k indices; this overlap is not evidence of shared influential groups.'},
        'scope': 'All-five-correct Belebele subset only. No correctness-stratified, layerwise, variance-decomposition, or latent-country extension is performed here.',
    }
    (OUT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    stats.write_csv(OUT / 'language_pairs.csv', rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()

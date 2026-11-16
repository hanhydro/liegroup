# 1DSfM Ellis Island inputs

Source: Wilson and Snavely, "Robust Global Translations with 1DSfM", ECCV 2014,
https://www.cs.cornell.edu/projects/1dsfm/

The raw scene is not included here because the dataset has no stated license. Run the
following command from the top folder to download it from the benchmark website and verify it.

    bash data/fetch_1dsfm.sh

The program places two files in `data/Ellis_Island/` and checks them against `SHA256SUMS`.

- `EGs.txt` holds the pairwise relative rotations.
- `gt_bundle.out` holds the Bundler reconstruction that serves as the public reference frame.

To verify a copy obtained earlier, run `shasum -a 256 -c SHA256SUMS` from `data/`.

`code/real_rotation_averaging.py` selects the target cameras by public degree, builds their
incident edges, applies the public-frame filter, and writes `results/real_ra_results.npz` and
`results/real_ra_edge_records.json`.

set -euo pipefail
cd "$(dirname "$0")"
URL="https://landmark.cs.cornell.edu/projects/1dsfm/datasets.tar.gz"
if [ ! -f Ellis_Island/EGs.txt ] || [ ! -f Ellis_Island/gt_bundle.out ]; then
  echo "Downloading 1DSfM numerical data from $URL ..."
  tb="$(mktemp -t 1dsfm.XXXXXX)"
  if command -v curl >/dev/null 2>&1; then curl -fL -o "$tb" "$URL"
  elif command -v wget >/dev/null 2>&1; then wget -O "$tb" "$URL"
  else echo "Need curl or wget on PATH." >&2; exit 1; fi
  td="$(mktemp -d)"; tar -xzf "$tb" -C "$td"
  mkdir -p Ellis_Island
  for f in EGs.txt gt_bundle.out; do
    src="$(find "$td" -type f -path '*Ellis_Island*'"/$f" | head -1)"
    [ -n "$src" ] || { echo "Ellis_Island/$f not found in the downloaded archive." >&2; exit 1; }
    cp "$src" "Ellis_Island/$f"
  done
  rm -rf "$td" "$tb"
fi
echo "Verifying against SHA256SUMS ..."
shasum -a 256 -c SHA256SUMS
echo "OK: 1DSfM Ellis_Island inputs verified."

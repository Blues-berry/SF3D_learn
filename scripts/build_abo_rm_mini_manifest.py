from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "abo_rm_mini"
BUCKET_HTTP_ROOT = "https://amazon-berkeley-objects.s3.amazonaws.com"
BUCKET_API_ROOT = "https://amazon-berkeley-objects.s3.amazonaws.com/"
LISTINGS_PREFIX = "listings/metadata/"
MODELS_KEY = "3dmodels/metadata/3dmodels.csv.gz"
DEFAULT_DOWNLOAD_CONNECT_TIMEOUT = 30
DEFAULT_DOWNLOAD_READ_TIMEOUT = 180
DEFAULT_DOWNLOAD_RETRIES = 6
DEFAULT_DOWNLOAD_BACKOFF_SECONDS = 2.0
PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]
RISK_BUCKETS = [
    "over_smoothing",
    "metal_nonmetal_confusion",
    "local_highlight_misread",
    "boundary_bleed",
]
VIEWS = [
    "front_studio",
    "three_quarter_indoor",
    "side_neon",
]

RISK_KEYWORDS = {
    "over_smoothing": {
        "fabric",
        "faux linen",
        "linen",
        "textile",
        "woven",
        "weave",
        "rattan",
        "wicker",
        "leather",
        "microfiber",
        "velvet",
        "suede",
        "tufted",
        "upholster",
        "upholstered",
        "wood",
        "wooden",
        "grain",
        "bamboo",
        "rope",
        "jute",
        "knit",
    },
    "metal_nonmetal_confusion": {
        "metal",
        "steel",
        "iron",
        "brass",
        "bronze",
        "chrome",
        "copper",
        "nickel",
        "aluminum",
        "alloy",
        "gold",
        "silver",
        "stainless",
        "galvanized",
        "mirror",
        "mirrored",
    },
    "local_highlight_misread": {
        "glass",
        "mirror",
        "mirrored",
        "ceramic",
        "glossy",
        "gloss",
        "polished",
        "marble",
        "glazed",
        "lacquer",
        "shiny",
        "reflective",
        "crystal",
        "acrylic",
        "porcelain",
        "chrome",
        "brass",
    },
    "boundary_bleed": {
        "frame",
        "framed",
        "mirror",
        "lamp",
        "stool",
        "chair",
        "desk",
        "table",
        "shelf",
        "cabinet",
        "glass",
        "metal",
        "wood",
        "base",
        "leg",
        "handle",
        "edge",
        "trim",
        "door",
        "drawer",
        "stand",
    },
}
NONMETAL_KEYWORDS = {
    "wood",
    "fabric",
    "leather",
    "linen",
    "textile",
    "ceramic",
    "glass",
    "acrylic",
    "plastic",
    "rattan",
    "wicker",
    "marble",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quota-per-bucket", type=int, default=50)
    parser.add_argument("--manifest-name", type=str, default="objects_200.json")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-connect-timeout", type=int, default=DEFAULT_DOWNLOAD_CONNECT_TIMEOUT)
    parser.add_argument("--download-read-timeout", type=int, default=DEFAULT_DOWNLOAD_READ_TIMEOUT)
    parser.add_argument("--download-retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES)
    parser.add_argument("--download-backoff-seconds", type=float, default=DEFAULT_DOWNLOAD_BACKOFF_SECONDS)
    parser.add_argument("--allow-env-proxy", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def make_http_session(allow_env_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = allow_env_proxy
    return session


def describe_proxy_mode(allow_env_proxy: bool) -> dict[str, object]:
    configured = {key: os.environ[key] for key in PROXY_ENV_KEYS if os.environ.get(key)}
    return {
        "allow_env_proxy": allow_env_proxy,
        "active_env_proxy_keys": sorted(configured.keys()),
        "active_env_proxy_values": configured,
    }


def http_get_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[int, int],
    retries: int,
    backoff_seconds: float,
    stream: bool = False,
) -> requests.Response:
    errors = []
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            message = f"attempt {attempt}/{retries}: {exc}"
            errors.append(message)
            if attempt == retries:
                break
            sleep_seconds = min(backoff_seconds * (2 ** (attempt - 1)), 30.0)
            print(f"[download retry] {url} :: {message}; sleeping {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Download failed for {url}: {' | '.join(errors)}")


def fetch_s3_keys(
    prefix: str,
    *,
    session: requests.Session,
    timeout: tuple[int, int],
    retries: int,
    backoff_seconds: float,
) -> list[str]:
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys: list[str] = []
    continuation_token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            params["continuation-token"] = continuation_token
        query = urlencode(params)
        response = http_get_with_retry(
            session,
            f"{BUCKET_API_ROOT}?{query}",
            timeout=timeout,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        root = ElementTree.fromstring(response.text)
        keys.extend(elem.text for elem in root.findall("s3:Contents/s3:Key", ns) if elem.text)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns) or "").lower() == "true"
        continuation_token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        if not truncated or not continuation_token:
            break
    return keys


def download_cached(
    key: str,
    cache_dir: Path,
    *,
    session: requests.Session,
    timeout: tuple[int, int],
    retries: int,
    backoff_seconds: float,
    force: bool = False,
) -> Path:
    local_path = cache_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        return local_path
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    response = http_get_with_retry(
        session,
        f"{BUCKET_HTTP_ROOT}/{key}",
        timeout=timeout,
        retries=retries,
        backoff_seconds=backoff_seconds,
        stream=True,
    )
    try:
        with response, tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(local_path)
    except Exception:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return local_path


def read_gzip_jsonl(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_gzip_csv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_value(entries) -> str:
    if not entries:
        return ""
    if isinstance(entries, str):
        return entries
    if isinstance(entries, list):
        english = None
        first = None
        for entry in entries:
            if isinstance(entry, str):
                first = first or entry
                if first:
                    break
            elif isinstance(entry, dict):
                value = entry.get("value")
                if value:
                    first = first or value
                    if entry.get("language_tag", "").lower().startswith("en"):
                        english = value
                        break
        return english or first or ""
    if isinstance(entries, dict):
        return str(entries.get("value") or "")
    return str(entries)


def collect_values(entries) -> list[str]:
    values = []
    if not entries:
        return values
    if isinstance(entries, str):
        return [entries]
    if isinstance(entries, dict):
        value = entries.get("value")
        return [str(value)] if value else []
    for entry in entries:
        if isinstance(entry, str):
            values.append(entry)
        elif isinstance(entry, dict):
            value = entry.get("value")
            if value:
                values.append(str(value))
    return values


def collect_node_paths(entries) -> list[str]:
    paths = []
    for entry in entries or []:
        if isinstance(entry, dict):
            path = entry.get("path")
            if path:
                paths.append(str(path))
    return paths


def normalize_text(values: Iterable[str]) -> list[str]:
    seen = set()
    items = []
    for value in values:
        text = " ".join(str(value).strip().split())
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        items.append(text)
    return items


def contains_keyword(text_blob: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text_blob
    pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
    return re.search(pattern, text_blob) is not None


def score_keyword_set(text_blob: str, keywords: set[str]) -> tuple[float, list[str]]:
    hits = sorted(keyword for keyword in keywords if contains_keyword(text_blob, keyword))
    return float(len(hits)), hits


def float_or_zero(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def int_or_zero(value) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def category_from_listing(product_type: str, node_paths: list[str]) -> str:
    if product_type:
        return product_type
    if not node_paths:
        return "unknown"
    leaf = node_paths[0].split("/")[-1].strip()
    return leaf or "unknown"


def sample_group_from_listing(node_paths: list[str], category: str) -> str:
    if node_paths:
        parts = [part for part in node_paths[0].split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        if parts:
            return parts[-1]
    return category or "unknown"


def dedupe_priority(record: dict) -> tuple:
    metadata_richness = sum(
        1
        for key in ["item_name", "product_type", "category", "node_paths", "material_values", "finish_values"]
        if record.get(key)
    )
    return (
        metadata_richness,
        len(record.get("material_values", [])),
        len(record.get("node_paths", [])),
    )


def build_candidate(listing: dict, model_meta: dict) -> dict:
    item_name = first_value(listing.get("item_name"))
    product_type = first_value(listing.get("product_type"))
    material_values = normalize_text(collect_values(listing.get("material")))
    fabric_values = normalize_text(collect_values(listing.get("fabric_type")))
    finish_values = normalize_text(collect_values(listing.get("finish_type")))
    node_paths = normalize_text(collect_node_paths(listing.get("node")))
    category = category_from_listing(product_type, node_paths)

    text_fields = normalize_text(
        [
            item_name,
            product_type,
            first_value(listing.get("product_description")),
            *material_values,
            *fabric_values,
            *finish_values,
            *node_paths,
        ]
    )
    text_blob = " | ".join(text_fields).lower()

    metal_score, metal_hits = score_keyword_set(text_blob, RISK_KEYWORDS["metal_nonmetal_confusion"])
    highlight_score, highlight_hits = score_keyword_set(text_blob, RISK_KEYWORDS["local_highlight_misread"])
    over_score, over_hits = score_keyword_set(text_blob, RISK_KEYWORDS["over_smoothing"])
    boundary_score, boundary_hits = score_keyword_set(text_blob, RISK_KEYWORDS["boundary_bleed"])
    nonmetal_score, nonmetal_hits = score_keyword_set(text_blob, NONMETAL_KEYWORDS)

    materials_count = int_or_zero(model_meta.get("materials"))
    meshes_count = int_or_zero(model_meta.get("meshes"))
    textures_count = int_or_zero(model_meta.get("textures"))

    mixed_material_bonus = 0.8 if metal_score > 0 and nonmetal_score > 0 else 0.0
    multi_part_bonus = 0.3 if materials_count >= 2 else 0.0
    texture_bonus = 0.25 if textures_count >= 3 else 0.0
    mesh_bonus = 0.25 if meshes_count >= 2 else 0.0

    risk_scores = {
        "over_smoothing": round(
            over_score + (multi_part_bonus + texture_bonus if over_score > 0 else 0.0),
            3,
        ),
        "metal_nonmetal_confusion": round(
            metal_score + (mixed_material_bonus + multi_part_bonus if metal_score > 0 else 0.0),
            3,
        ),
        "local_highlight_misread": round(
            highlight_score + (texture_bonus + 0.25 * (metal_score > 0) if highlight_score > 0 else 0.0),
            3,
        ),
        "boundary_bleed": round(
            boundary_score + (multi_part_bonus + mesh_bonus + 0.25 * (metal_score > 0) if boundary_score > 0 else 0.0),
            3,
        ),
    }
    total_risk = round(sum(risk_scores.values()), 3)

    return {
        "id": str(listing.get("3dmodel_id")),
        "item_id": str(listing.get("item_id") or listing.get("3dmodel_id")),
        "label": slugify(item_name or f"{category}_{listing.get('3dmodel_id')}"),
        "name": item_name or str(listing.get("3dmodel_id")),
        "product_type": product_type,
        "category": category,
        "sample_group": sample_group_from_listing(node_paths, category),
        "node_paths": node_paths,
        "material_values": material_values,
        "fabric_values": fabric_values,
        "finish_values": finish_values,
        "path": model_meta["path"],
        "model_meta": {
            "meshes": meshes_count,
            "materials": materials_count,
            "textures": textures_count,
            "images": int_or_zero(model_meta.get("images")),
            "vertices": int_or_zero(model_meta.get("vertices")),
            "faces": int_or_zero(model_meta.get("faces")),
            "extent_x": float_or_zero(model_meta.get("extent_x")),
            "extent_y": float_or_zero(model_meta.get("extent_y")),
            "extent_z": float_or_zero(model_meta.get("extent_z")),
        },
        "risk_scores": risk_scores,
        "risk_signals": {
            "over_smoothing": over_hits,
            "metal_nonmetal_confusion": metal_hits + nonmetal_hits,
            "local_highlight_misread": highlight_hits,
            "boundary_bleed": boundary_hits,
        },
        "risk_total": total_risk,
        "source_text": text_fields,
        "views": VIEWS,
    }


def interleave_by_group(candidates: list[dict], bucket: str) -> list[dict]:
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["sample_group"]].append(candidate)

    for group in groups.values():
        group.sort(
            key=lambda item: (
                -item["risk_scores"][bucket],
                -item["risk_total"],
                -item["model_meta"]["materials"],
                item["id"],
            )
        )

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    )

    interleaved = []
    keep_going = True
    while keep_going:
        keep_going = False
        for _, group in ordered_groups:
            if group:
                interleaved.append(group.pop(0))
                keep_going = True
    return interleaved


def choose_bucket_order(quotas: dict[str, int], pools: dict[str, list[dict]], used: set[str]) -> list[str]:
    availabilities = []
    for bucket in RISK_BUCKETS:
        if quotas[bucket] <= 0:
            continue
        remaining = sum(1 for candidate in pools[bucket] if candidate["id"] not in used)
        availabilities.append((remaining, bucket))
    availabilities.sort()
    return [bucket for _, bucket in availabilities]


def choose_candidate(pool: list[dict], used: set[str], recent_groups: dict[str, str], bucket: str):
    for candidate in pool:
        if candidate["id"] in used:
            continue
        if recent_groups.get(bucket) == candidate["sample_group"]:
            continue
        return candidate
    for candidate in pool:
        if candidate["id"] not in used:
            return candidate
    return None


def sample_manifest(candidates: list[dict], quota_per_bucket: int, target_size: int) -> tuple[list[dict], dict]:
    quotas = {bucket: quota_per_bucket for bucket in RISK_BUCKETS}
    pools = {
        bucket: interleave_by_group(
            [candidate for candidate in candidates if candidate["risk_scores"][bucket] > 0.0],
            bucket,
        )
        for bucket in RISK_BUCKETS
    }
    used_ids = set()
    recent_groups: dict[str, str] = {}
    selected = []
    picks_by_bucket = defaultdict(int)

    progress = True
    while progress and any(quotas[bucket] > 0 for bucket in RISK_BUCKETS):
        progress = False
        for bucket in choose_bucket_order(quotas, pools, used_ids):
            candidate = choose_candidate(pools[bucket], used_ids, recent_groups, bucket)
            if candidate is None:
                continue
            selected_record = dict(candidate)
            selected_record["sampling_bucket"] = bucket
            selected_record["sampling_score"] = candidate["risk_scores"][bucket]
            selected.append(selected_record)
            used_ids.add(candidate["id"])
            recent_groups[bucket] = candidate["sample_group"]
            quotas[bucket] -= 1
            picks_by_bucket[bucket] += 1
            progress = True
            if len(selected) >= target_size:
                break
        if len(selected) >= target_size:
            break

    if len(selected) < target_size:
        remaining = [candidate for candidate in candidates if candidate["id"] not in used_ids]
        remaining.sort(
            key=lambda item: (
                -item["risk_total"],
                -max(item["risk_scores"].values()),
                item["id"],
            )
        )
        for candidate in remaining:
            selected_record = dict(candidate)
            selected_record["sampling_bucket"] = max(
                RISK_BUCKETS,
                key=lambda bucket: (candidate["risk_scores"][bucket], -RISK_BUCKETS.index(bucket)),
            )
            selected_record["sampling_score"] = candidate["risk_scores"][selected_record["sampling_bucket"]]
            selected.append(selected_record)
            used_ids.add(candidate["id"])
            picks_by_bucket["backfill"] += 1
            if len(selected) >= target_size:
                break

    summary = {
        "requested_target_size": target_size,
        "requested_quota_per_bucket": quota_per_bucket,
        "selected_size": len(selected),
        "bucket_pick_counts": dict(picks_by_bucket),
        "remaining_quotas": quotas,
        "candidate_count": len(candidates),
        "positive_pool_sizes": {bucket: len(pool) for bucket, pool in pools.items()},
    }
    return selected[:target_size], summary


def main():
    args = parse_args()
    if args.target_size != args.quota_per_bucket * len(RISK_BUCKETS):
        raise ValueError(
            "This mini benchmark assumes equal quotas across the 4 fixed risk buckets. "
            "Please keep target-size == quota-per-bucket * 4."
        )

    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir or (output_dir / "cache"))
    metadata_cache_dir = ensure_dir(cache_dir / "metadata")
    rng = random.Random(args.seed)
    download_timeout = (args.download_connect_timeout, args.download_read_timeout)
    download_session = make_http_session(allow_env_proxy=args.allow_env_proxy)

    model_metadata_path = download_cached(
        MODELS_KEY,
        metadata_cache_dir,
        session=download_session,
        timeout=download_timeout,
        retries=args.download_retries,
        backoff_seconds=args.download_backoff_seconds,
        force=args.force_download,
    )
    model_rows = read_gzip_csv(model_metadata_path)
    model_by_id = {row["3dmodel_id"]: row for row in model_rows if row.get("3dmodel_id") and row.get("path")}

    listing_keys = fetch_s3_keys(
        LISTINGS_PREFIX,
        session=download_session,
        timeout=download_timeout,
        retries=args.download_retries,
        backoff_seconds=args.download_backoff_seconds,
    )
    deduped_candidates: dict[str, dict] = {}
    skipped = defaultdict(list)

    for key in listing_keys:
        listing_path = download_cached(
            key,
            metadata_cache_dir,
            session=download_session,
            timeout=download_timeout,
            retries=args.download_retries,
            backoff_seconds=args.download_backoff_seconds,
            force=args.force_download,
        )
        for listing in read_gzip_jsonl(listing_path):
            model_id = listing.get("3dmodel_id")
            if not model_id:
                skipped["missing_3dmodel_id"].append({"item_id": listing.get("item_id")})
                continue
            model_meta = model_by_id.get(str(model_id))
            if model_meta is None:
                skipped["missing_3dmodel_metadata"].append({"3dmodel_id": model_id})
                continue
            candidate = build_candidate(listing, model_meta)
            if not candidate["name"] or candidate["name"] == candidate["id"]:
                skipped["missing_item_name"].append({"3dmodel_id": model_id})
                continue
            existing = deduped_candidates.get(candidate["id"])
            if existing is None or dedupe_priority(candidate) > dedupe_priority(existing):
                deduped_candidates[candidate["id"]] = candidate

    candidates = list(deduped_candidates.values())
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda item: (
            -item["risk_total"],
            -max(item["risk_scores"].values()),
            item["id"],
        )
    )

    selected, summary = sample_manifest(
        candidates,
        quota_per_bucket=args.quota_per_bucket,
        target_size=args.target_size,
    )

    manifest_path = output_dir / args.manifest_name
    manifest_payload = {
        "meta": {
            "source": "Amazon Berkeley Objects",
            "target_size": args.target_size,
            "quota_per_bucket": args.quota_per_bucket,
            "seed": args.seed,
            "views": VIEWS,
            "sampling_strategy": "risk-stratified round-robin over four fixed failure-risk buckets with category interleaving",
        },
        "objects": selected,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    bucket_distribution = defaultdict(int)
    for item in selected:
        bucket_distribution[item["sampling_bucket"]] += 1

    debug_payload = {
        "summary": summary,
        "selected_bucket_distribution": dict(bucket_distribution),
        "total_candidates_after_dedup": len(candidates),
        "listing_files": listing_keys,
        "skipped_counts": {reason: len(rows) for reason, rows in skipped.items()},
        "download_config": {
            "connect_timeout": args.download_connect_timeout,
            "read_timeout": args.download_read_timeout,
            "retries": args.download_retries,
            "backoff_seconds": args.download_backoff_seconds,
            **describe_proxy_mode(args.allow_env_proxy),
        },
    }
    (output_dir / "manifest_build_report.json").write_text(json.dumps(debug_payload, indent=2))
    (output_dir / "manifest_skips.json").write_text(json.dumps(skipped, indent=2))

    print(f"Wrote manifest with {len(selected)} objects to {manifest_path}")
    print(f"Build report: {output_dir / 'manifest_build_report.json'}")
    print(f"Skip log: {output_dir / 'manifest_skips.json'}")


if __name__ == "__main__":
    main()

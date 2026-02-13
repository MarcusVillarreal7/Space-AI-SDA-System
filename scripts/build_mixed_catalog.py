#!/usr/bin/env python3
"""
Build a mixed space object catalog from multiple TLE sources.

Merges payloads (active satellites), debris, and rocket bodies into a single
1000-object catalog with realistic composition (~60% payload, ~35% debris,
~5% rocket body). Produces a merged TLE file, type manifest, and optionally
generates the simulation dataset with object_type embedded in the parquet.

Usage:
    python scripts/build_mixed_catalog.py --seed 42 --generate
    python scripts/build_mixed_catalog.py --payloads 600 --debris 350 --rocket-bodies 50 --generate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.tle_loader import TLELoader, TLE
from src.utils.logging_config import get_logger

logger = get_logger("build_catalog")

# Object type constants
PAYLOAD = "PAYLOAD"
DEBRIS = "DEBRIS"
ROCKET_BODY = "ROCKET_BODY"

# TLE sources by object type
PAYLOAD_SOURCES = ["active"]
DEBRIS_SOURCES = ["cosmos-2251-deb", "fengyun-1c-deb", "iridium-33-deb", "indian-asat-deb"]
# last-30-days contains a mix; we classify by name (R/B → ROCKET_BODY, DEB → DEBRIS, else PAYLOAD)

# Scenario-injector slots (objects at these indices must be PAYLOAD for injection to work)
SCENARIO_SLOTS = set(range(990, 997))


def classify_tle_type(tle: TLE, source_file: str) -> str:
    """
    Classify a TLE as PAYLOAD, DEBRIS, or ROCKET_BODY based on provenance
    and name heuristics.

    Priority:
        1. Source file: debris TLE files → DEBRIS
        2. Name contains 'DEB' → DEBRIS
        3. Name contains 'R/B' → ROCKET_BODY
        4. Everything else → PAYLOAD
    """
    source_stem = Path(source_file).stem.lower()

    # Debris source files
    if any(tag in source_stem for tag in ("deb", "debris")):
        return DEBRIS

    name_upper = tle.name.upper()

    if " DEB" in name_upper or name_upper.startswith("DEB "):
        return DEBRIS
    if "R/B" in name_upper or " RB" in name_upper:
        return ROCKET_BODY

    return PAYLOAD


def load_all_tles(raw_dir: Path) -> dict[str, list[tuple[TLE, str]]]:
    """
    Load TLEs from all available .tle files and classify them by type.

    Returns:
        Dict mapping object type to list of (TLE, source_file) tuples.
    """
    loader = TLELoader()
    by_type: dict[str, list[tuple[TLE, str]]] = {
        PAYLOAD: [],
        DEBRIS: [],
        ROCKET_BODY: [],
    }

    tle_files = sorted(raw_dir.glob("*.tle"))
    if not tle_files:
        raise FileNotFoundError(
            f"No TLE files found in {raw_dir}. "
            "Run: python scripts/download_tle_data.py --categories active cosmos-2251-deb fengyun-1c-deb iridium-33-deb last-30-days"
        )

    seen_catalog_numbers: set[int] = set()

    for tle_file in tle_files:
        try:
            tles = loader.load_from_file(tle_file)
        except Exception as e:
            logger.warning("Failed to load %s: %s", tle_file, e)
            continue

        for tle in tles:
            # Deduplicate by NORAD catalog number
            if tle.catalog_number in seen_catalog_numbers:
                continue
            seen_catalog_numbers.add(tle.catalog_number)

            obj_type = classify_tle_type(tle, str(tle_file))
            by_type[obj_type].append((tle, str(tle_file)))

    for t, items in by_type.items():
        logger.info("Loaded %d %s TLEs", len(items), t)

    return by_type


def sample_catalog(
    by_type: dict[str, list[tuple[TLE, str]]],
    n_payloads: int = 600,
    n_debris: int = 350,
    n_rocket_bodies: int = 50,
    seed: int = 42,
) -> list[tuple[TLE, str]]:
    """
    Sample a mixed catalog with the specified composition.

    Returns:
        Ordered list of (TLE, object_type) tuples, length = sum of counts.
        Indices 990-996 are guaranteed to be PAYLOAD (scenario injector slots).
    """
    rng = np.random.default_rng(seed)
    total = n_payloads + n_debris + n_rocket_bodies

    # Sample from each pool
    def _sample(pool: list, n: int, label: str) -> list[tuple[TLE, str]]:
        if len(pool) < n:
            logger.warning(
                "Only %d %s TLEs available, requested %d — using all",
                len(pool), label, n,
            )
            selected = pool.copy()
        else:
            indices = rng.choice(len(pool), size=n, replace=False)
            selected = [pool[i] for i in sorted(indices)]
        return [(tle, label) for tle, _src in selected]

    payloads = _sample(by_type[PAYLOAD], n_payloads, PAYLOAD)
    debris = _sample(by_type[DEBRIS], n_debris, DEBRIS)
    rocket_bodies = _sample(by_type[ROCKET_BODY], n_rocket_bodies, ROCKET_BODY)

    # Merge and shuffle (but we'll fix scenario slots afterward)
    merged = payloads + debris + rocket_bodies
    indices = list(range(len(merged)))
    rng.shuffle(indices)
    catalog = [merged[i] for i in indices]

    # Ensure indices 990-996 are PAYLOAD (swap if needed)
    payload_indices = [i for i, (_, t) in enumerate(catalog) if t == PAYLOAD and i not in SCENARIO_SLOTS]
    for slot in sorted(SCENARIO_SLOTS):
        if slot >= len(catalog):
            break
        if catalog[slot][1] != PAYLOAD:
            # Find a payload not in a scenario slot to swap with
            if payload_indices:
                swap_idx = payload_indices.pop(0)
                catalog[slot], catalog[swap_idx] = catalog[swap_idx], catalog[slot]
            else:
                logger.warning("Not enough PAYLOADs to fill scenario slot %d", slot)

    return catalog[:total]


def write_merged_tle(catalog: list[tuple[TLE, str]], output_path: Path) -> None:
    """Write the merged catalog as a standard 3-line TLE file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for tle, _ in catalog:
        lines.append(tle.name)
        lines.append(tle.line1)
        lines.append(tle.line2)
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote merged TLE: %s (%d objects)", output_path, len(catalog))


def write_type_manifest(catalog: list[tuple[TLE, str]], output_path: Path) -> None:
    """Write object type manifest as JSON: {index: type, ...}."""
    manifest = {str(i): obj_type for i, (_, obj_type) in enumerate(catalog)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote type manifest: %s", output_path)


def add_object_type_column(
    parquet_path: Path,
    manifest_path: Path,
    n_objects: int,
) -> None:
    """
    Post-process ground_truth.parquet to add an object_type column.

    Reads the type manifest and injects the column based on object_id → index
    mapping. The parquet is sorted by (object_id, time), so each object has
    n_timesteps contiguous rows.
    """
    import pandas as pd

    logger.info("Adding object_type column to %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Map object_id to type via index (object_ids are assigned 0..N-1 in order)
    unique_ids = sorted(df["object_id"].unique())
    id_to_type = {}
    for i, oid in enumerate(unique_ids):
        id_to_type[oid] = manifest.get(str(i), PAYLOAD)

    df["object_type"] = df["object_id"].map(id_to_type).fillna(PAYLOAD)
    df.to_parquet(parquet_path, index=False)

    # Verify
    type_counts = df.groupby("object_type")["object_id"].nunique()
    logger.info("Object type distribution in parquet:")
    for t, c in type_counts.items():
        logger.info("  %s: %d objects", t, c)


def main():
    parser = argparse.ArgumentParser(
        description="Build a mixed space object catalog (payloads + debris + rocket bodies)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--payloads", type=int, default=600, help="Number of payload objects")
    parser.add_argument("--debris", type=int, default=350, help="Number of debris objects")
    parser.add_argument("--rocket-bodies", type=int, default=50, help="Number of rocket body objects")
    parser.add_argument("--raw-dir", type=Path, default=project_root / "data" / "raw",
                        help="Directory containing TLE files")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "processed" / "ml_train_1k",
                        help="Output directory for generated dataset")
    parser.add_argument("--generate", action="store_true",
                        help="Also generate the simulation dataset (propagate orbits, measurements)")
    parser.add_argument("--no-type-column", action="store_true",
                        help="Skip adding object_type column to parquet")
    args = parser.parse_args()

    total = args.payloads + args.debris + args.rocket_bodies
    logger.info(
        "Building mixed catalog: %d PAYLOAD + %d DEBRIS + %d ROCKET_BODY = %d total (seed=%d)",
        args.payloads, args.debris, args.rocket_bodies, total, args.seed,
    )

    # Step 1: Load all available TLEs
    by_type = load_all_tles(args.raw_dir)

    # Step 2: Sample
    catalog = sample_catalog(
        by_type,
        n_payloads=args.payloads,
        n_debris=args.debris,
        n_rocket_bodies=args.rocket_bodies,
        seed=args.seed,
    )

    # Step 3: Write merged TLE + type manifest
    merged_tle_path = args.raw_dir / "mixed_catalog.tle"
    manifest_path = args.raw_dir / "mixed_catalog_types.json"
    write_merged_tle(catalog, merged_tle_path)
    write_type_manifest(catalog, manifest_path)

    # Step 4: Generate dataset if requested
    if args.generate:
        logger.info("Generating simulation dataset from mixed catalog...")
        from src.utils.config_loader import SimulationConfig
        from src.simulation.data_generator import DatasetGenerator

        config = SimulationConfig(
            num_objects=total,
            output_dir=args.output_dir,
        )
        generator = DatasetGenerator(config)
        dataset = generator.generate(tle_file=merged_tle_path, seed=args.seed)
        dataset.save(args.output_dir)
        logger.info("Dataset saved to %s", args.output_dir)

        # Step 5: Add object_type column
        if not args.no_type_column:
            parquet_path = args.output_dir / "ground_truth.parquet"
            if parquet_path.exists():
                add_object_type_column(parquet_path, manifest_path, total)
    else:
        logger.info(
            "Merged TLE written. To generate dataset, re-run with --generate, or run:\n"
            "  python -m src.simulation.data_generator --tle-file %s --output-dir %s",
            merged_tle_path, args.output_dir,
        )

    # Summary
    type_counts = {}
    for _, obj_type in catalog:
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
    logger.info("Catalog summary: %s", type_counts)

    # Verify scenario slots
    for slot in sorted(SCENARIO_SLOTS):
        if slot < len(catalog):
            _, t = catalog[slot]
            assert t == PAYLOAD, f"Slot {slot} must be PAYLOAD, got {t}"
    logger.info("Scenario slots 990-996: all PAYLOAD (verified)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

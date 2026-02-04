#!/usr/bin/env python3
"""
Download TLE (Two-Line Element) data from CelesTrak.
This data is used for simulating realistic satellite orbits.
"""

import sys
from pathlib import Path
import requests
from typing import List

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import get_logger

logger = get_logger("data_download")


# CelesTrak TLE sources
TLE_SOURCES = {
    "stations": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "active": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "analyst": "https://celestrak.org/NORAD/elements/gp.php?GROUP=analyst&FORMAT=tle",
    "weather": "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle",
    "noaa": "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=tle",
    "goes": "https://celestrak.org/NORAD/elements/gp.php?GROUP=goes&FORMAT=tle",
    "resource": "https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=tle",
    "sarsat": "https://celestrak.org/NORAD/elements/gp.php?GROUP=sarsat&FORMAT=tle",
    "dmc": "https://celestrak.org/NORAD/elements/gp.php?GROUP=dmc&FORMAT=tle",
    "tdrss": "https://celestrak.org/NORAD/elements/gp.php?GROUP=tdrss&FORMAT=tle",
    "geo": "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle",
}


def download_tle(url: str, output_path: Path) -> bool:
    """
    Download TLE data from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save TLE data
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response.text)
        
        # Count TLEs (each TLE is 3 lines: name + 2 data lines)
        lines = response.text.strip().split('\n')
        num_tles = len(lines) // 3
        
        logger.info(f"Downloaded {num_tles} TLEs to {output_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False


def download_all(categories: List[str] = None, output_dir: Path = None):
    """
    Download TLE data for specified categories.
    
    Args:
        categories: List of category names to download (None = all)
        output_dir: Output directory (default: data/raw)
    """
    if output_dir is None:
        output_dir = project_root / "data" / "raw"
    
    if categories is None:
        categories = list(TLE_SOURCES.keys())
    
    logger.info(f"Downloading TLE data for categories: {categories}")
    
    success_count = 0
    fail_count = 0
    
    for category in categories:
        if category not in TLE_SOURCES:
            logger.warning(f"Unknown category: {category}")
            continue
        
        url = TLE_SOURCES[category]
        output_path = output_dir / f"{category}.tle"
        
        if download_tle(url, output_path):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Download complete: {success_count} succeeded, {fail_count} failed")
    
    if success_count > 0:
        logger.info(f"TLE files saved to: {output_dir}")
        logger.info("You can now run simulations using this data")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download TLE data from CelesTrak")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(TLE_SOURCES.keys()),
        help="Categories to download (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available TLE categories:")
        for category in TLE_SOURCES.keys():
            print(f"  - {category}")
        return 0
    
    try:
        download_all(args.categories, args.output_dir)
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

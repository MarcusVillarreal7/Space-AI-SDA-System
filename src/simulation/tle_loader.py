"""
TLE (Two-Line Element) data loading and parsing utilities.
Handles loading TLE data from files and CelesTrak.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import requests

from src.utils.logging_config import get_logger

logger = get_logger("simulation")


@dataclass
class TLE:
    """Two-Line Element set for a satellite."""
    
    name: str
    line1: str
    line2: str
    catalog_number: int
    epoch: float
    
    @classmethod
    def from_lines(cls, name: str, line1: str, line2: str) -> "TLE":
        """
        Create TLE from three lines of text.
        
        Args:
            name: Satellite name
            line1: First line of TLE
            line2: Second line of TLE
        
        Returns:
            TLE object
        
        Example:
            >>> name = "ISS (ZARYA)"
            >>> line1 = "1 25544U 98067A   ..."
            >>> line2 = "2 25544  51.6461 ..."
            >>> tle = TLE.from_lines(name, line1, line2)
        """
        # Extract catalog number from line 1 (columns 3-7)
        catalog_number = int(line1[2:7])
        
        # Extract epoch from line 1 (columns 19-32)
        epoch = float(line1[18:32])
        
        return cls(
            name=name.strip(),
            line1=line1.strip(),
            line2=line2.strip(),
            catalog_number=catalog_number,
            epoch=epoch
        )
    
    def __repr__(self) -> str:
        return f"TLE(name='{self.name}', catalog={self.catalog_number}, epoch={self.epoch:.2f})"


class TLELoader:
    """Load and manage TLE data from various sources."""
    
    CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
    
    def __init__(self):
        """Initialize TLE loader."""
        self.tles: List[TLE] = []
        logger.info("TLE loader initialized")
    
    def load_from_file(self, filepath: Path) -> List[TLE]:
        """
        Load TLE data from a file.
        
        Args:
            filepath: Path to TLE file (3-line format)
        
        Returns:
            List of TLE objects
        
        Example:
            >>> loader = TLELoader()
            >>> tles = loader.load_from_file("data/raw/active.tle")
            >>> print(f"Loaded {len(tles)} TLEs")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"TLE file not found: {filepath}")
            raise FileNotFoundError(f"TLE file not found: {filepath}")
        
        logger.info(f"Loading TLEs from {filepath}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        tles = []
        
        # Parse 3-line format (name, line1, line2)
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            
            # Validate TLE format
            if line1.startswith('1 ') and line2.startswith('2 '):
                try:
                    tle = TLE.from_lines(name, line1, line2)
                    tles.append(tle)
                except Exception as e:
                    logger.warning(f"Failed to parse TLE for {name}: {e}")
                    continue
        
        self.tles = tles
        logger.info(f"Loaded {len(tles)} TLEs from {filepath}")
        
        return tles
    
    def load_from_celestrak(self, category: str = "active") -> List[TLE]:
        """
        Load TLE data from CelesTrak.
        
        Args:
            category: CelesTrak category (e.g., 'active', 'stations', 'weather')
        
        Returns:
            List of TLE objects
        
        Note:
            Requires internet connection
        """
        url = f"{self.CELESTRAK_BASE_URL}?GROUP={category}&FORMAT=tle"
        
        logger.info(f"Downloading TLEs from CelesTrak: {category}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            tles = []
            
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break
                
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                if line1.startswith('1 ') and line2.startswith('2 '):
                    try:
                        tle = TLE.from_lines(name, line1, line2)
                        tles.append(tle)
                    except Exception as e:
                        logger.warning(f"Failed to parse TLE for {name}: {e}")
                        continue
            
            self.tles = tles
            logger.info(f"Downloaded {len(tles)} TLEs from CelesTrak")
            
            return tles
            
        except requests.RequestException as e:
            logger.error(f"Failed to download TLEs from CelesTrak: {e}")
            raise
    
    def filter_by_altitude(self, min_alt_km: float = 200, max_alt_km: float = 2000) -> List[TLE]:
        """
        Filter TLEs by approximate altitude range.
        
        Args:
            min_alt_km: Minimum altitude (km)
            max_alt_km: Maximum altitude (km)
        
        Returns:
            Filtered list of TLEs
        
        Note:
            Uses mean motion to estimate altitude (approximate)
        """
        filtered = []
        
        for tle in self.tles:
            # Extract mean motion from line 2 (revs per day)
            mean_motion = float(tle.line2[52:63])
            
            # Approximate altitude from mean motion
            # Using simplified formula: a = (μ/(2πn/86400)^2)^(1/3) - R_earth
            # where μ = 398600.4418 km³/s², R_earth = 6378.137 km
            n_rad_per_sec = mean_motion * 2 * 3.14159265359 / 86400
            semi_major_axis = (398600.4418 / (n_rad_per_sec ** 2)) ** (1/3)
            altitude = semi_major_axis - 6378.137
            
            if min_alt_km <= altitude <= max_alt_km:
                filtered.append(tle)
        
        logger.info(f"Filtered {len(filtered)}/{len(self.tles)} TLEs by altitude ({min_alt_km}-{max_alt_km} km)")
        
        return filtered
    
    def get_by_catalog_number(self, catalog_number: int) -> Optional[TLE]:
        """
        Get TLE by NORAD catalog number.
        
        Args:
            catalog_number: NORAD catalog number
        
        Returns:
            TLE object or None if not found
        """
        for tle in self.tles:
            if tle.catalog_number == catalog_number:
                return tle
        
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded TLEs.
        
        Returns:
            Dictionary with statistics
        """
        if not self.tles:
            return {"count": 0}
        
        epochs = [tle.epoch for tle in self.tles]
        
        return {
            "count": len(self.tles),
            "min_epoch": min(epochs),
            "max_epoch": max(epochs),
            "avg_epoch": sum(epochs) / len(epochs),
        }


# Example usage
if __name__ == "__main__":
    # Test TLE loading
    loader = TLELoader()
    
    # Try loading from file if it exists
    tle_file = Path("data/raw/active.tle")
    if tle_file.exists():
        tles = loader.load_from_file(tle_file)
        print(f"Loaded {len(tles)} TLEs")
        
        if tles:
            print(f"First TLE: {tles[0]}")
            
        stats = loader.get_statistics()
        print(f"Statistics: {stats}")
    else:
        print(f"TLE file not found: {tle_file}")
        print("Run: python scripts/download_tle_data.py")

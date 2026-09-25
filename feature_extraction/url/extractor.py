import sys
import os

# Robust import handling to allow running as a module or as a standalone script
try:
    from .lexical import LexicalFeatureExtractor
    from .entropy import EntropyFeatureExtractor
    from .statistics import StatisticalFeatureExtractor
except ImportError:
    # Fallback for direct execution
    from lexical import LexicalFeatureExtractor
    from entropy import EntropyFeatureExtractor
    from statistics import StatisticalFeatureExtractor

class URLExtractor:
    """
    Master Extractor for all URL-based features. 
    It acts as an orchestrator, delegating specific feature extraction tasks 
    to specialized sub-modules (Lexical, Entropy, Statistics) and merging their outputs.
    """
    def __init__(self, url: str):
        self.url = url.strip()

    def extract(self) -> dict:
        """
        Compile all extracted URL features into a single, flattened dictionary
        ready for the AI Reasoning Engine.
        """
        # 1. Initialize the combined features dictionary
        combined_features = {}

        # 2. Get Lexical Features (String characteristics, keywords, lengths)
        try:
            lexical_extractor = LexicalFeatureExtractor(self.url)
            combined_features.update(lexical_extractor.extract_features())
        except Exception as e:
            print(f"Error extracting lexical features: {str(e)}")

        # 3. Get Entropy Features (Statistical randomness of domain, path, query)
        try:
            entropy_extractor = EntropyFeatureExtractor(self.url)
            combined_features.update(entropy_extractor.extract_features())
        except Exception as e:
            print(f"Error extracting entropy features: {str(e)}")

        # 4. Get Statistical Features (Subdomains, ratios, consecutive characters)
        try:
            stats_extractor = StatisticalFeatureExtractor(self.url)
            combined_features.update(stats_extractor.extract_features())
        except Exception as e:
            print(f"Error extracting statistical features: {str(e)}")
        
        return combined_features
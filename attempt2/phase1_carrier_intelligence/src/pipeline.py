"""
pipeline.py

Carrier Intelligence Pipeline

Executes the complete Phase-1 workflow:

1. Load Model
2. Extract Weights
3. Layer Analysis
4. Statistical Analysis
5. Entropy Analysis
6. Importance Analysis
7. Chaotic Carrier Selection
8. Carrier Ranking
Project: Neural-Entropic Steganography (NES v2)
"""

from pathlib import Path
import logging

from model_loader import ModelLoader
from weight_extractor import WeightExtractor
from layer_analyzer import LayerAnalyzer
from statistics import StatisticsAnalyzer
from entropy import EntropyAnalyzer
from importance import ImportanceAnalyzer
from chaotic_selector import ChaoticSelector
from carrier_ranker import CarrierRanker


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


class CarrierIntelligencePipeline:

    def __init__(self):

        self.output_dir = Path("../outputs")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.loader = None
        self.model = None

    ####################################################################
    # Step 1
    ####################################################################

    def load_model(self):

        logging.info("=" * 80)
        logging.info("STEP 1 : Loading Model")
        logging.info("=" * 80)

        self.loader = ModelLoader()

        _, self.model = self.loader.load()

    ####################################################################
    # Step 2
    ####################################################################

    def extract_weights(self):

        logging.info("=" * 80)
        logging.info("STEP 2 : Weight Extraction")
        logging.info("=" * 80)

        extractor = WeightExtractor(self.model)

        metadata = extractor.extract()

        extractor.export(metadata)

    ####################################################################
    # Step 3
    ####################################################################

    def layer_analysis(self):

        logging.info("=" * 80)
        logging.info("STEP 3 : Layer Analysis")
        logging.info("=" * 80)

        analyzer = LayerAnalyzer(self.model)

        results = analyzer.analyze()

        analyzer.export(results)

    ####################################################################
    # Step 4
    ####################################################################

    def statistical_analysis(self):

        logging.info("=" * 80)
        logging.info("STEP 4 : Statistical Analysis")
        logging.info("=" * 80)

        analyzer = StatisticsAnalyzer(self.model)

        results = analyzer.analyze()

        analyzer.export(results)

    ####################################################################
    # Step 5
    ####################################################################

    def entropy_analysis(self):

        logging.info("=" * 80)
        logging.info("STEP 5 : Entropy Analysis")
        logging.info("=" * 80)

        analyzer = EntropyAnalyzer(self.model)

        results = analyzer.analyze()

        analyzer.export(results)

    ####################################################################
    # Step 6
    ####################################################################

    def importance_analysis(self):

        logging.info("=" * 80)
        logging.info("STEP 6 : Importance Analysis")
        logging.info("=" * 80)

        analyzer = ImportanceAnalyzer(self.model)

        results = analyzer.analyze()

        analyzer.export(results)

    ####################################################################
    # Step 7
    ####################################################################

    def chaotic_selection(self):

        logging.info("=" * 80)
        logging.info("STEP 7 : Chaotic Carrier Selection")
        logging.info("=" * 80)

        selector = ChaoticSelector(self.model)

        results = selector.select()

        selector.export(results)

    ####################################################################
    # Step 8
    ####################################################################

    def rank_carriers(self):

        logging.info("=" * 80)
        logging.info("STEP 8 : Carrier Ranking")
        logging.info("=" * 80)

        ranker = CarrierRanker()

        ranking = ranker.rank()

        ranker.export(ranking)

        print()
        print("=" * 80)
        print("TOP 10 CARRIER LAYERS")
        print("=" * 80)
        print()

        print(
            ranking[
                [
                    "rank",
                    "layer_name",
                    "final_score"
                ]
            ].head(10)
        )

    ####################################################################
    # Execute Pipeline
    ####################################################################

    def run(self):

        logging.info("")
        logging.info("=" * 80)
        logging.info("NES PHASE 1 : CARRIER INTELLIGENCE")
        logging.info("=" * 80)

        self.load_model()

        self.extract_weights()

        self.layer_analysis()

        self.statistical_analysis()

        self.entropy_analysis()

        self.importance_analysis()

        self.chaotic_selection()

        self.rank_carriers()

        logging.info("")
        logging.info("=" * 80)
        logging.info("PHASE 1 COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)


if __name__ == "__main__":

    pipeline = CarrierIntelligencePipeline()

    pipeline.run()
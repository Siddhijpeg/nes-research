import json
from pathlib import Path

from src.residuals.extractor import load_tinyllama, TensorExtractor
from src.profiling.statistics import TensorStatistics
from src.profiling.layer_profiler import LayerProfiler
from src.profiling.report_generator import ReportGenerator

OUTPUT_DIR = Path("data/profiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():

    print("Loading TinyLlama...")

    model = load_tinyllama()

    extractor = TensorExtractor(model)

    tensors = extractor.extract_target_tensors()

    profiles = {}

    total = len(tensors)

    for idx, (name, tensor) in enumerate(tensors.items(), start=1):

        print(f"[{idx}/{total}] Profiling {name}")

        profiles[name] = TensorStatistics.compute(tensor)

    output_file = OUTPUT_DIR / "tinyllama_weight_profile.json"

    with open(output_file, "w") as f:
        json.dump(profiles, f, indent=2)

    print("\nSaved:")
    print(output_file)

    family_summary = LayerProfiler.aggregate(output_file)
    family_file = OUTPUT_DIR / "tensor_family_summary.json"

    with open(family_file, "w") as f:
        json.dump(family_summary, f, indent=2)

    print(f"Saved: {family_file}")

    plots_dir = Path(
        "experiments/exp01_residual_characterization/plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = ReportGenerator.load_summary(
        family_file
    )

    ReportGenerator.create_bar_chart(
        summary,
        "std",
        plots_dir / "std_by_family.png"
    )

    ReportGenerator.create_bar_chart(
        summary,
        "variance",
        plots_dir / "variance_by_family.png"
    )

    ReportGenerator.create_bar_chart(
        summary,
        "entropy",
        plots_dir / "entropy_by_family.png"
    )

    ReportGenerator.create_bar_chart(
        summary,
        "kurtosis",
        plots_dir / "kurtosis_by_family.png"
    )

    print("\nPlots generated.")


if __name__ == "__main__":
    main()
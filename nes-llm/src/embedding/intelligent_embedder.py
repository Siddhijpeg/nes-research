from copy import deepcopy

import torch

from src.embedding.embedding_result import (
    EmbeddingResult,
    LayerEmbeddingMetadata,
)

from src.profiling.model_profiler import (
    ModelProfiler,
)

from src.carrier_intelligence.feature_extractor import (
    CarrierFeatureExtractor,
)

from src.carrier_intelligence.quality_score import (
    QualityScore,
)

from src.carrier_intelligence.layer_profiler import (
    LayerProfiler,
)

from src.carrier_intelligence.carrier_scheduler import (
    CarrierScheduler,
)

from src.carrier_intelligence.carrier_score import (
    CarrierScore,
)

from src.carrier_intelligence.confidence_estimator import (
    ConfidenceEstimator,
)

from src.carrier_intelligence.adaptive_margin import (
    AdaptiveMarginController,
)

from src.carrier_intelligence.selector import (
    CarrierSelector,
)

from src.carrier_intelligence.layer_importance import LayerImportance

from src.carrier_intelligence.carrier_reliability import CarrierReliability

from src.carrier_intelligence.feature_normalizer import FeatureNormalizer
from src.carrier_intelligence.local_entropy import LocalEntropyEstimator

class IntelligentEmbedder:
    """
    Intelligent Carrier Embedding Engine.

    This class orchestrates the complete
    Carrier Intelligence Framework.

    Pipeline

        ModelProfiler

            ↓

        Feature Extraction

            ↓

        Quality Score

            ↓

        Layer Profiling

            ↓

        Payload Scheduling

            ↓

        Adaptive Margin

            ↓

        Carrier Selection

            ↓

        Residual Embedding

            ↓

        EmbeddingResult
    """

    def __init__(

        self,

        alpha=0.25,

    ):

        self.profiler = (
            ModelProfiler()
        )

        self.extractor = (
            CarrierFeatureExtractor()
        )

        self.quality = (
            QualityScore()
        )

        self.layer_profiler = (
            LayerProfiler()
        )

        self.scheduler = (
            CarrierScheduler()
        )

        self.confidence = (
            ConfidenceEstimator()
        )

        self.carrier_score = (
            CarrierScore()
        )

        self.margin = (
            AdaptiveMarginController(
                alpha=alpha,
            )
        )

        self.selector = (
            CarrierSelector()
        )

        self.layer_importance = LayerImportance()

        self.reliability = CarrierReliability()

        self.feature_extractor = CarrierFeatureExtractor()
        self.feature_normalizer = FeatureNormalizer()
        self.carrier_score = CarrierScore()

        self.quality_score = QualityScore()

        self.local_entropy = LocalEntropyEstimator()
        self.layer_importance = LayerImportance()
        self.reliability = CarrierReliability()

        self.scheduler = CarrierScheduler()

    ##################################################################
    # PRIVATE HELPERS
    ##################################################################

    def _validate_payload(
        self,
        payload_bits,
    ):

        if len(payload_bits) == 0:

            raise ValueError(
                "Payload is empty."
            )

    def _clone_profiles(
        self,
        profiles,
    ):

        cloned = []

        for profile in profiles:

            cloned.append(

                {

                    "layer":
                    profile["layer"],

                    "module":
                    profile["module"],

                    "fp16":
                    profile["fp16"],

                    "nf4":
                    profile["nf4"],

                    "residual":
                    profile["residual"].clone(),

                }

            )

        return cloned

    def _compute_quality(

        self,

        residual,

        fp16,

        nf4,

    ):

        features = self.extractor.extract(

            residual,

            fp16,

            nf4,

        )

        quality = self.quality.compute(

            features

        )

        return quality

    def _compute_carrier_score(

        self,

        residual,

        fp16_weight,

        quantized_weight,

    ):

        features = self.feature_extractor.extract(
            residual,
            fp16_weight,
            quantized_weight,
        )

        ##########################################################
        # Additional Feature Columns
        ##########################################################

        quality = self.quality.compute(features)

        quality = quality.reshape(residual.shape)

        print("Residual shape :", residual.shape)
        print("Quality shape  :", quality.shape)

        margin = self.margin.compute(
            residual,
            quality,
        )

        print("Margin shape   :", margin.shape)

        confidence = self.confidence.compute(
            residual,
            margin,
        )

        ##########################################################
        # Append Features
        ##########################################################

        features = torch.cat(
            [
                features,
                quality.reshape(-1, 1),
                margin.reshape(-1, 1),
                confidence.reshape(-1, 1),
            ],
            dim=1,
        )
        ##########################################################
        # Local Entropy Feature
        ##########################################################

        entropy = self.local_entropy.compute(
            residual
        )

        features = torch.cat(

            [

                features,

                entropy.reshape(-1, 1),

            ],

            dim=1,

        )

        ##########################################################
        # Layer Importance Feature
        ##########################################################

        importance = self.layer_importance.compute(
            fp16_weight
        )

        features = torch.cat(

            [

                features,

                importance.reshape(-1, 1),

            ],

            dim=1,

        )

        ##########################################################
        # Carrier Reliability
        ##########################################################

        reliability = self.reliability.compute(
            features
        )

        features = torch.cat(

            [

                features,

                reliability.reshape(-1,1),

            ],

            dim=1,

        )

        ##########################################################
        # Normalize
        ##########################################################

        features = self.feature_normalizer.normalize(
            features
        )

        ##########################################################
        # Carrier Utility Estimator
        ##########################################################

        score = self.carrier_score.compute(
            features
        )

        score = score.reshape(
            residual.shape
        )

        return score, quality, confidence, margin

    def _profile_layers(
        self,
        profiles,
    ):

        summaries = []

        for profile in profiles:

            quality = self._compute_quality(

                profile["residual"],

                profile["fp16"],

                profile["nf4"],

            )

            summary = self.layer_profiler.profile(
                quality
            )

            summary["layer"] = profile["layer"]

            summary["module"] = profile["module"]

            summaries.append(
                summary
            )

        return summaries

    ##################################################################
    # MAIN EMBEDDING ENTRY
    ##################################################################

    def embed(

        self,

        fp16_model,

        nf4_model,

        payload_bits,

    ):

        ##############################################################
        # Validate Payload
        ##############################################################

        self._validate_payload(
            payload_bits
        )

        ##############################################################
        # Build Residual Profiles
        ##############################################################

        profiles = self.profiler.profile(

            fp16_model,

            nf4_model,

        )

        ##############################################################
        # Clone Profiles
        ##############################################################

        profiles = self._clone_profiles(
            profiles
        )

        ##############################################################
        # Carrier Scores
        ##############################################################

        carrier_profiles = []

        carrier_scores = []

        for profile in profiles:

            score, quality, confidence, margins = (

                self._compute_carrier_score(

                    profile["residual"],

                    profile["fp16"],

                    profile["nf4"],

                )

            )

            profile["quality"] = quality
            profile["confidence"] = confidence
            profile["margins"] = margins

            ##########################################################
            # Robust Profile Score
            ##########################################################

            flat_score = score.flatten()

            k = max(

                1,

                int(

                    0.05 * flat_score.numel()

                ),

            )

            profile_score = torch.topk(

                flat_score,

                k,

            ).values.mean()

            carrier_profiles.append(
                profile
            )

            carrier_scores.append(
                profile_score
            )

        carrier_scores = torch.tensor(
            carrier_scores
        )

        ##############################################################
        # Scheduler
        ##############################################################

        allocation_plan = self.scheduler.allocate(

            carrier_profiles,

            carrier_scores,

            len(payload_bits),

        )

        print("\n===== Scheduler =====")

        print("Allocations:", len(allocation_plan))

        active = 0
        bits = 0

        for a in allocation_plan:

            if a.allocated_bits > 0:

                active += 1
                bits += a.allocated_bits

        print("Active carriers:", active)
        print("Allocated bits:", bits)

        print("=====================\n")

        ##############################################################
        # Containers
        ##############################################################

        stego_profiles = []

        layer_metadata = []

        ##############################################################
        # Start Embedding
        ##############################################################

        cursor = 0

        for allocation in allocation_plan:

            layer_id = allocation.layer

            module = allocation.module

            n_bits = allocation.allocated_bits

            bits = payload_bits[
                cursor:
                cursor + n_bits
            ]

            cursor += n_bits

            if len(bits) == 0:
                continue

            if len(bits) == 0:

                continue

            ##########################################################
            # Locate corresponding residual profile
            ##########################################################

            candidate_profiles = [

                p

                for p in profiles

                if

                p["layer"] == layer_id

                and

                p["module"] == module

            ]

            print(
                f"[MATCH] layer={layer_id} "
                f"module={module} "
                f"matches={len(candidate_profiles)}"
            )

            ##########################################################
            # Embed inside every module of this layer
            ##########################################################

            for profile in candidate_profiles:

                residual = profile["residual"]

                fp16 = profile["fp16"]

                nf4 = profile["nf4"]

                module = profile["module"]

                ##########################################################
                # Quality Scores
                ##########################################################

                quality_scores = self._compute_quality(

                    residual,

                    fp16,

                    nf4,

                )

                ##########################################################
                # Adaptive Margins
                ##########################################################

                margins = self.margin.compute(

                    residual,

                    quality_scores,

                )

                ##########################################################
                # Carrier Selection
                ##########################################################

                capacity = residual.numel()

                bits_to_embed = min(

                    len(bits),

                    capacity,

                )

                positions = self.selector.select(

                    residual,

                    fp16,

                    nf4,

                    bits_to_embed,

                )

                ##########################################################
                # Clone Residual
                ##########################################################

                stego = residual.clone()

                flat = stego.flatten()

                margin_flat = margins.flatten()

                ##########################################################
                # Sign Embedding
                ##########################################################

                for bit, pos in zip(

                    bits[:bits_to_embed],

                    positions,

                ):

                    delta = margin_flat[pos]

                    value = abs(

                        flat[pos]

                    )

                    if bit == 1:

                        flat[pos] = (

                            value

                            +

                            delta

                        )

                    else:

                        flat[pos] = -(

                            value

                            +

                            delta

                        )

                ##########################################################
                # Save Embedded Profile
                ##########################################################

                profile["residual"] = stego

                print(
                    f"[EMBED] layer={layer_id} "
                    f"module={module} "
                    f"bits={bits_to_embed}"
                )

                stego_profiles.append(

                    {

                        "layer":

                        layer_id,

                        "module":

                        module,

                        "fp16":

                        fp16,

                        "nf4":

                        nf4,

                        "residual":

                        stego,

                    }

                )

                ##########################################################
                # Save Metadata
                ##########################################################

                layer_metadata.append(

                    LayerEmbeddingMetadata(

                        layer=layer_id,

                        module=module,

                        positions=positions,

                        margins=margins,

                        quality_scores=quality_scores,

                        payload_size=bits_to_embed,

                        capacity=capacity,

                    )

                )

        print(
            f"[ALLOC] layer={allocation.layer} "
            f"module={allocation.module} "
            f"bits={allocation.allocated_bits}"
        )
        
        ##############################################################
        # Validate
        ##############################################################

        if len(stego_profiles) == 0:

            raise RuntimeError(

                "No residuals were embedded."

            )

        ##############################################################
        # Return
        ##############################################################

        return EmbeddingResult(

            stego_profiles=stego_profiles,

            allocation_plan=allocation_plan,

            layer_metadata=layer_metadata,

        )
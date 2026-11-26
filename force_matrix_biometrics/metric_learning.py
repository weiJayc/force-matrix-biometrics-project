"""Metric learning for the pressure CNN: an ArcFace embedding and a cross-day contrastive loss.

Instead of only learning to pick one of the training people, the model learns an embedding
where each person's presses sit close together and away from everyone else, so a distance
threshold can reject someone it never saw. Requires PyTorch; cnn.py imports this module only
when --loss arcface is used.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

FEATURE_CHANNELS = 32  # output size of cnn._feature_extractor


class PressureArcFaceNet(nn.Module):
    """Same convolutional features as the classifier, then a unit-length embedding.

    Each training person has a learned direction (class_weight). forward() returns
    scale * cosine to each direction, so it can be used like the classifier's logits.
    """

    def __init__(self, features: nn.Module, class_count: int, embedding_dim: int, scale: float, margin: float) -> None:
        super().__init__()
        self.features = features
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(FEATURE_CHANNELS, embedding_dim),
        )
        self.class_weight = nn.Parameter(torch.empty(class_count, embedding_dim))
        nn.init.xavier_uniform_(self.class_weight)
        self.scale = scale
        self.margin = margin

    def embed(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(self.features(tensor)), dim=1)

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings @ F.normalize(self.class_weight, dim=1).T

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.scale * self.cosine(self.embed(tensor))

    def margin_logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ArcFace training logits: the true person's angle is widened by `margin` radians,
        so a press only scores well when it sits clearly closer to its own direction."""
        cosine = self.cosine(embeddings).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine * cosine)
        widened = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        # Beyond an angle of pi - margin, cos(angle + margin) would start rising again; use a
        # linear penalty there instead (the standard ArcFace fallback).
        widened = torch.where(
            cosine > math.cos(math.pi - self.margin), widened, cosine - math.sin(math.pi - self.margin) * self.margin
        )
        target = F.one_hot(labels, cosine.size(1)).bool()
        return self.scale * torch.where(target, widened, cosine)


def cross_day_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, days: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Supervised contrastive loss whose positives are the same person on a DIFFERENT day.

    For each press, other presses by the same person from another day are pulled closer, and
    presses by other people (any day) are pushed away. Presses by the same person on the same
    day are left out of both, so the model gains nothing from matching session conditions.
    Presses with no other-day partner in the batch do not contribute.
    """
    similarity = embeddings @ embeddings.T / temperature
    same_person = labels[:, None] == labels[None, :]
    same_day = days[:, None] == days[None, :]
    positive = same_person & ~same_day
    candidates = positive | ~same_person
    similarity = similarity.masked_fill(~candidates, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    has_positive = positive_count > 0
    if not has_positive.any():
        return embeddings.sum() * 0.0
    per_press = -log_probability.masked_fill(~positive, 0.0).sum(dim=1) / positive_count.clamp_min(1)
    return per_press[has_positive].mean()


class PersonDayBatchSampler:
    """Batches of `samples_per_person` presses from every training person, spread over that
    person's recording days in turn, so every batch has cross-day partners to compare."""

    def __init__(
        self, labels: list[int], days: list[int], samples_per_person: int, batches_per_epoch: int, seed: int
    ) -> None:
        self.pools: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, day) in enumerate(zip(labels, days)):
            self.pools[label][day].append(index)
        self.samples_per_person = samples_per_person
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)
        self.queues: dict[tuple[int, int], list[int]] = {}

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _draw(self, label: int, day: int) -> int:
        """Next press from this person-day; each one is used once before any repeats."""
        queue = self.queues.get((label, day))
        if not queue:
            queue = self.pools[label][day][:]
            self.rng.shuffle(queue)
            self.queues[(label, day)] = queue
        return queue.pop()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.batches_per_epoch):
            batch = []
            for label in sorted(self.pools):
                person_days = sorted(self.pools[label])
                self.rng.shuffle(person_days)
                batch.extend(
                    self._draw(label, person_days[i % len(person_days)]) for i in range(self.samples_per_person)
                )
            yield batch


def build_arcface_model(features: Any, class_count: int, embedding_dim: int, scale: float, margin: float) -> nn.Module:
    return PressureArcFaceNet(features, class_count, embedding_dim, scale, margin)

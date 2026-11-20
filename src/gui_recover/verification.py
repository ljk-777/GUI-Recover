"""Visual-only action-outcome interfaces and small, auditable classifiers.

No environment, teacher, reward, injection or split metadata enters this API.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from PIL import Image

from .actions import Action, parse_action
from .input_guard import inspect_input


STATUSES = ("achieved", "not_achieved", "unknown")


@dataclass(frozen=True, slots=True)
class TransitionView:
    before: Image.Image
    after: Image.Image
    proposed: Action
    goal: str
    target: str
    phase: str = "immediate"

    def __post_init__(self):
        parse_action(json.dumps(self.proposed.to_dict()))
        if self.phase not in ("immediate", "after_wait"):
            raise ValueError("Unknown observation phase")
        if not isinstance(self.goal,str) or not isinstance(self.target,str) or not self.goal or not self.target:
            raise ValueError("Visual verification requires a user goal and current target")


def pixel_hash(picture):
    picture = picture.convert("RGB")
    return hashlib.sha256(str(picture.size).encode() + picture.tobytes()).hexdigest()


def observation_key(view):
    # Used by OFFLINE observability auditing only, never as a runtime label lookup.
    return hashlib.sha256(json.dumps([pixel_hash(view.before), pixel_hash(view.after),
        view.proposed.to_dict(), view.goal, view.target, view.phase], sort_keys=True).encode()).hexdigest()


def pixel_features(view):
    """Fixed global and action-local features, without file names or task IDs."""
    arrays = [np.asarray(p.convert("RGB").resize((12, 12)), dtype=np.float64).reshape(-1) / 255
              for p in (view.before, view.after)]
    before, after = arrays
    x, y = view.proposed.position or (0.5, 0.5)
    crops = []
    for p in (view.before, view.after):
        w, h = p.size
        xx, yy = x * w, y * h
        crop = p.convert("L").crop((int(xx)-48, int(yy)-24, int(xx)+48, int(yy)+24)).resize((12, 6))
        crops.append(np.asarray(crop, dtype=np.float64).reshape(-1) / 255)
    action = [float(view.proposed.action == kind) for kind in ("CLICK", "INPUT", "HOVER", "TYPE")]
    action += [x, y, min(len(view.proposed.value or ""), 100) / 100, float(view.phase == "after_wait")]
    return np.concatenate([before, after, after-before, np.abs(after-before),
                           crops[0], crops[1], crops[1]-crops[0], np.asarray(action)])


def appearance_verdict(view):
    if view.before.size != view.after.size:
        return {"status": "unknown", "reason": "size_changed"}
    if view.proposed.action == "INPUT":
        result = inspect_input(view.before, view.after, view.proposed.position, version="v2")
        status = {"text_visible": "achieved", "empty": "not_achieved"}.get(result["status"], "unknown")
        return {"status": status, "reason": "appearance_only_not_content_correctness", "signal": result}
    difference = float(np.abs(np.asarray(view.before, dtype=float)-np.asarray(view.after, dtype=float)).mean()/255)
    return {"status": "not_achieved" if difference <= 0.002 else "achieved",
            "reason": "pixel_change_is_not_semantic_success", "difference": difference}


def _softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=1, keepdims=True)


def fit_classifier(features, labels, weights, *, l2, rank=16, steps=600):
    """Train-only PCA and a three-class linear head; fixed full-batch optimizer."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray([STATUSES.index(x) for x in labels])
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    if len(features) < 3 or features.ndim != 2 or len(set(labels)) < 2:
        raise ValueError("Classifier needs multiple classes and nonempty training data")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-3)
    z = (features-mean)/scale
    # Gram formulation avoids a large d x d covariance matrix.
    values, vectors = np.linalg.eigh(z @ z.T / len(z))
    kept = np.argsort(values)[::-1][:min(rank, len(z)-1)]
    kept = kept[values[kept] > 1e-8]
    components = (z.T @ vectors[:, kept]) / np.sqrt(len(z)*values[kept])[None, :]
    projected = z @ components
    projection_scale = np.maximum(projected.std(axis=0), 1e-3)
    direct_count = min(8,features.shape[1])
    inputs = np.column_stack([projected/projection_scale, z[:,-direct_count:], np.ones(len(z))])
    coefficients = np.zeros((inputs.shape[1], 3))
    targets = np.eye(3)[labels]
    for _ in range(steps):
        probabilities = _softmax(inputs @ coefficients)
        gradient = inputs.T @ ((probabilities-targets)*weights[:, None])
        gradient[:-1] += l2 * coefficients[:-1]
        coefficients -= 0.1 * gradient
    return {"mean": mean, "scale": scale, "components": components,
            "projection_scale": projection_scale, "coefficients": coefficients,
            "direct_count":np.asarray(direct_count)}


def probabilities(model, features):
    z = (np.asarray(features)-model["mean"])/model["scale"]
    projected = (z @ model["components"])/model["projection_scale"]
    direct_count=int(model["direct_count"])
    return _softmax(np.column_stack([projected,z[:,-direct_count:],np.ones(len(z))]) @ model["coefficients"])


def confidence_verdict(probability, threshold):
    probability = np.asarray(probability)
    if (probability.shape != (3,) or not np.isfinite(probability).all() or
            (probability<0).any() or (probability>1).any() or abs(probability.sum()-1)>1e-5):
        raise ValueError("Expected three finite status probabilities")
    chosen = int(probability.argmax())
    status = STATUSES[chosen] if probability[chosen] >= threshold else "unknown"
    return {"status": status, "probabilities": dict(zip(STATUSES, probability.tolist())),
            "threshold": float(threshold)}

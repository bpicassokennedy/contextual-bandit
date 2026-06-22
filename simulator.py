import json
import numpy as np

with open("image_features.json") as f:
    image_features = json.load(f)

with open("personas.json") as f:
    personas = json.load(f)["personas"]

NOISE_STD = 0.3

def featurize(img):
    """Convert raw image features dict into a numeric vector matching persona weight keys."""
    return {
        "num_people": min(img["num_people"] / 5.0, 1.0),
        "age_child": 1.0 if "child" in img.get("age_brackets", []) else 0.0,
        "age_adult": 1.0 if "adult" in img.get("age_brackets", []) else 0.0,
        "age_older_adult": 1.0 if "older_adult" in img.get("age_brackets", []) else 0.0,
        "setting_outdoor": 1.0 if img["setting"] == "outdoor" else 0.0,
        "setting_indoor": 1.0 if img["setting"] == "indoor" else 0.0,
        "color_warm": 1.0 if img["color_temperature"] == "warm" else 0.0,
        "color_cool": 1.0 if img["color_temperature"] == "cool" else 0.0,
        "color_neutral": 1.0 if img["color_temperature"] == "neutral" else 0.0,
        "brightness_bright": 1.0 if img["brightness"] == "bright" else 0.0,
        "brightness_moderate": 1.0 if img["brightness"] == "moderate" else 0.0,
        "brightness_dim": 1.0 if img["brightness"] == "dim" else 0.0,
        "nature_water": 1.0 if "water" in img.get("nature_present", []) else 0.0,
        "nature_plants": 1.0 if "plants" in img.get("nature_present", []) else 0.0,
        "nature_animals": 1.0 if "animals" in img.get("nature_present", []) else 0.0,
        "time_morning": 1.0 if img.get("time_of_day") == "morning" else 0.0,
        "time_afternoon": 1.0 if img.get("time_of_day") == "afternoon" else 0.0,
        "time_evening": 1.0 if img.get("time_of_day") in ("evening", "sunset", "dusk") else 0.0,
    }


def score(persona, img_features, rng=None):
    """Compute engagement score = dot(preferences, features) + noise, clipped to [0, 1]."""
    if rng is None:
        rng = np.random.default_rng()
    vec = featurize(img_features)
    weights = persona["preferences"]
    raw = sum(weights.get(k, 0.0) * v for k, v in vec.items())
    # scale into ~[0,1] range so scores don't all clip to 1.0
    scaled = (raw + 3.0) / 6.0
    noisy = scaled + rng.normal(0, NOISE_STD)
    return float(np.clip(noisy, 0.0, 1.0))


def get_feature_keys():
    """Return ordered list of feature keys for building numeric vectors."""
    return list(featurize(next(iter(image_features.values()))).keys())


def image_to_vector(img_features):
    """Convert image features to a numpy array in consistent key order."""
    vec = featurize(img_features)
    return np.array([vec[k] for k in get_feature_keys()])


def persona_to_vector(persona):
    """Convert persona preferences to a numpy array in consistent key order."""
    keys = get_feature_keys()
    return np.array([persona["preferences"].get(k, 0.0) for k in keys])


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Sanity Check: Top 3 images per persona ===\n")
    for persona in personas:
        scores = {}
        for filename, features in image_features.items():
            if "_parse_error" in features:
                continue
            avg = np.mean([score(persona, features, rng) for _ in range(50)])
            scores[filename] = avg

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        print(f"{persona['name']} ({persona['background'][:60]}...)")
        for fname, s in ranked[:3]:
            print(f"  {s:.3f}  {fname}")
        print()

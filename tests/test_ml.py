from ml.data import deterministic_split, load_synthetic_fixture
from ml.model import EchoSafeModel
from ml.tokenizer import WordTokenizer


def test_tokenizer_round_trip():
    tokenizer = WordTokenizer().fit(["happy calm", "sad alone"])
    ids = tokenizer.encode("happy alone", max_length=8)
    assert len(ids) == 8
    assert tokenizer.decode(ids) == "happy alone"


def test_model_forward_shapes():
    import torch

    model = EchoSafeModel(vocab_size=100, emotion_classes=7)
    ids = torch.randint(0, 100, (2, 3, 16))
    lengths = torch.tensor([3, 2])
    output = model(ids, lengths)
    assert output["emotion_probs"].shape == (2, 7)
    assert output["stress_prob"].shape == (2,)


def test_synthetic_fixture_and_split():
    records = load_synthetic_fixture("ml/fixtures/synthetic.jsonl")
    train, validation = deterministic_split(records, seed=42)
    assert records
    assert train


def test_real_go_emotions_and_dreaddit_checkpoints_load():
    import torch
    from ml.real_model import TextClassifier
    from ml.tokenizer import WordTokenizer

    go_checkpoint = "ml/checkpoints/real/go_emotions_cnn_bigru_attention.pt"
    go_vocab = "ml/experiments/go_emotions_cnn_bigru_attention/vocab.json"
    payload = torch.load(go_checkpoint, map_location="cpu", weights_only=True)
    model = TextClassifier(
        vocab=payload["model_config"]["vocab_size"],
        classes=28,
        architecture="cnn_bigru_attention",
    )
    model.load_state_dict(payload["model_state"], strict=True)
    tokenizer = WordTokenizer.load(go_vocab)
    ids = torch.tensor([tokenizer.encode("I feel happy today", max_length=96)])
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(ids))[0]
    assert probabilities.shape == (28,)

    stress_checkpoint = "ml/checkpoints/real/dreaddit_cnn_bigru_attention.pt"
    stress_vocab = "ml/experiments/dreaddit_cnn_bigru_attention/vocab.json"
    payload = torch.load(stress_checkpoint, map_location="cpu", weights_only=True)
    model = TextClassifier(
        vocab=payload["model_config"]["vocab_size"],
        classes=1,
        architecture="cnn_bigru_attention",
    )
    model.load_state_dict(payload["model_state"], strict=True)
    tokenizer = WordTokenizer.load(stress_vocab)
    ids = torch.tensor([tokenizer.encode("I am overwhelmed", max_length=96)])
    with torch.inference_mode():
        probability = torch.sigmoid(model(ids))[0, 0]
    assert 0.0 <= float(probability) <= 1.0

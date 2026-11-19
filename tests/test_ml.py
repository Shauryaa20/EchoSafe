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

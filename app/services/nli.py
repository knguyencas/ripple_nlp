from transformers import pipeline as hf_pipeline

_nli = None

def get_nli():
    global _nli
    if _nli is None:
        _nli = hf_pipeline(
            'zero-shot-classification',
            model='Xenova/nli-deberta-v3-small'
        )
    return _nli

def check_context(text: str) -> dict:
    nli = get_nli()
    hypotheses = [
        "This person is doing something for a normal everyday reason",
        "This person is saying goodbye or giving away belongings as farewell",
        "This person feels hopeless and does not want to continue living",
    ]
    result = nli(text, hypotheses, multi_label=True)
    scores = dict(zip(result['labels'], result['scores']))
    return {
        'is_grounded': scores[hypotheses[0]] > 0.6,
        'is_crisis':   scores[hypotheses[1]] > 0.5
                       or scores[hypotheses[2]] > 0.5,
        'scores':      scores,
    }
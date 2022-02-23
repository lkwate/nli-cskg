from typing import Dict, Any
from keybert import KeyBERT
import datasets
from functools import partial
from loguru import logger


def word_extraction(kw_model: KeyBERT, src: str):
    keywords = kw_model.extract_keywords(src)
    keywords = {item[0] for item in keywords}

    return keywords


def process_item(item: Dict[str, Any], kw_model: KeyBERT):
    premise = item["premise"]
    hypothesis = item["hypothesis"]
    keywords = word_extraction(kw_model, premise) | word_extraction(
        kw_model, hypothesis
    )
    item["keywords"] = keywords
    return item


def transform(data: datasets.arrow_dataset.Dataset, kw_model):
    transform_fn = partial(process_item, kw_model=kw_model)
    data = data.map(transform_fn)

    return data


if __name__ == "__main__":
    kw_model = KeyBERT()
    data = datasets.load_dataset("multi_nli")
    train = data["train"]
    val_matched = data["validation_matched"]
    val_mismatched = data["validation_mismatched"]

    dropped_columns = [
        "promptID",
        "premise_binary_parse",
        "premise_parse",
        "hypothesis_binary_parse",
        "hypothesis_parse",
        "genre",
    ]
    train = train.remove_columns(dropped_columns)
    val_matched = val_matched.remove_columns(dropped_columns)
    val_mismatched = val_mismatched.remove_columns(dropped_columns)

    logger.info("Transform training set")
    train = transform(train, kw_model)
    logger.info("Transform validation matched")
    val_matched = transform(val_matched, kw_model)
    logger.info("Transformer validation mismatched")
    val_mismatched = transform(val_mismatched, kw_model)

    data["train"], data["validation_matched"], data["validation_mismatched"] = (
        train,
        val_matched,
        val_mismatched,
    )
    data.save_to_disk("multi_nli_keywords")

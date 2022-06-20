import tensorflow_hub as hub
import tensorflow_text
import vlog

LOG = vlog.get_logger(__name__)
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"


def main():
    preprocesser = hub.KerasLayer(preprocess_url)
    text = ["nice movie indeed", "I love python programming"]
    text_preprocessed = preprocesser(text)
    print(text_preprocessed.keys())
    print(text_preprocessed["input_type_ids"])


if __name__ == "__main__":
    main()

## Taken from https://github.com/stefan-it/turkish-bert/blob/master/CHEATSHEET.md

from utils.file_utils import read_json_file

tokenizer_config = read_json_file("../configs/tokenizer.json")

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=tokenizer_config["lowercase"],
)

trainer = tokenizer.train(
    tokenizer_config["input_file"],
    vocab_size=tokenizer_config["vocabulary_size"],
    min_frequency=tokenizer_config["minimum_frequency"],
    show_progress=True,
    special_tokens=tokenizer_config["special_tokens"],
    limit_alphabet=tokenizer_config["alphabet_limit"],
    wordpieces_prefix="##"
)

tokenizer.save_model(tokenizer_config["output_path"], tokenizer_config["output_name"])
tokenizer.save("{}/{}.json".format(tokenizer_config["output_path"], tokenizer_config["output_name"]), pretty=True)
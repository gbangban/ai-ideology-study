# Student module


def fix_mistral_tokenizer(tokenizer):
    """Apply the Mistral regex fix to a tokenizer manually.

    Replicates what transformers does when fix_mistral_regex=True is passed
    to AutoTokenizer.from_pretrained(), since Unsloth doesn't support that kwarg.
    """
    import tokenizers

    setattr(tokenizer, "fix_mistral_regex", True)
    split_pretokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ),
        behavior="isolated",
    )
    current_pretokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
        tokenizer.backend_tokenizer.pre_tokenizer[0] = split_pretokenizer
    else:
        if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Metaspace):
            current_pretokenizer = tokenizers.pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False
            )
        tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [split_pretokenizer, current_pretokenizer]
        )

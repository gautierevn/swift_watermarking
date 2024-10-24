source venv/bin/activate # add your venv here

python watermark_dir.py -L facebook/opt-125m --watermark_power 1.0 --modulation cyclic --adapter_path models/finetuned_llm \
    --watermark_encoder_model models/Hide-R/encoder.pth --watermark_decoder_model models/Hide-R/decoder.pth
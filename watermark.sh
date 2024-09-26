source /home/gevennou/ip2p/bin/activate #USE MODIFED TRANSFORMERS PACKAGE

python /home/gevennou/swift/watermark_dir.py -L facebook/opt-125m --watermark_power 1.0 --modulation cyclic --adapter_path models/finetuned_llm/ \
    --watermark_encoder_model models/Hide-R/bzhenc.pth --watermark_decoder_model models/Hide-R/bzhdec.pth
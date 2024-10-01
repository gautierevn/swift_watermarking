source venv/bin/activate 

WATERMARKED_40_DB="watermarked_images/"
# BENIGN_40_DB = ""
# SYNTH_40_DB = ""
# OUTPUT_DIR = ""
transform_list=( "jpeg_compression" "grayscale" "resize_128" "noise" "crop_400" )

run_detection() {
    input_dir=$1
    output_file=$2
    python detect.py \
    --input_directory "$input_dir" \
    --output_file "$output_file" \
    --modulation "cyclic"
}

# Run for 40dB directories
run_detection "$WATERMARKED_40_DB" "watermarked_40db_results.json"
# run_detection "$BENIGN_40_DB" "benign_40db_results.json"
# run_detection "$SYNTH_40_DB" "synth_40db_results.json"


echo "All detection tasks completed."
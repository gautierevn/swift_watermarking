#OAR -n hider_42db
#OAR -q production
#OAR -p abacus21
#OAR -l gpu=1,walltime=20
#OAR -O /home/gevennou/output_batch_job/marking%jobid%.out.log
#OAR -E /home/gevennou/output_batch_job/marking%jobid%.err.log

source /home/gevennou/swift_release/bin/activate #USE MODIFED TRANSFORMERS PACKAGE

transform_list=( "jpeg_compression" "grayscale" "resize_128" "noise" "crop_400" )

WATERMARKED_40_DB="swift/watermarked_images/"
# BENIGN_40_DB = ""
# SYNTH_40_DB = ""
# OUTPUT_DIR = ""

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
# run_detection "$BENIGN_40_DB" "/home/gevennou/BIG_storage/semantic_watermarking/benign_40db_results.json"
# run_detection "$SYNTH_40_DB" "/home/gevennou/BIG_storage/semantic_watermarking/synth_40db_results.json"

# # Run for each transform in the transform_list
# for transform in "${transform_list[@]}"; do
#     echo "Running detection for $transform"
#     input_dir="${WATERMARKED_40_DB}${transform}/"
#     output_file="${OUTPUT_DIR}/${transform}_COCO_40db_results.json"
#     run_detection "$input_dir" "$output_file"
# done

echo "All detection tasks completed."
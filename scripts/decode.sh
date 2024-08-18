python decode_outputs.py \
    --vocab ./vocab \
    --sample_file ./trained_models/train-splitlora-gpt2sm-rank8/beam_prediction.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file ./ref_files/e2e_ref_rank8.txt \
    --output_pred_file ./ref_files/e2e_pred_rank8.txt
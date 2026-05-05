cd /path/to/distributed-inference
python run_sweep.py \
  --python python \
  --main_script main5.py \
  --input_file input.txt \
  --model_id /home/dingcong/models/google/gemma-2-2b-it \
  --output_dir sweep_outputs \
  --csv_file sweep_results.csv
如果你想单独跑一组（例如 bs=64, sl=256, group2）：

python main5.py \
  --model_id /home/dingcong/models/google/gemma-2-2b-it \
  --input_file input.txt \
  --output_file outputs_bs64_sl256.txt \
  --summary_file summary_bs64_sl256.txt \
  --batch_size 64 \
  --sequence_length 256 \
  --num_cpu_layers 3 \
  --mid_gpu_layer 12 \
  --build_dataset
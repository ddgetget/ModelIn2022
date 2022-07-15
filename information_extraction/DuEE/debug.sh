

dataset_name=DuEE-Fin
data_dir=~/data/DuEE-fin
conf_dir=conf/DuEE-fin
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=~/data/DuEE-fin/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=2
epoch=1

model=trigger
is_train=True
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_sequence_labeling.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}

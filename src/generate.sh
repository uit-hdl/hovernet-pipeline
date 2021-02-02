# Model type that will train or run inference
profile=consep
export H_PROFILE=hv_${profile}

python create_config.py \
    --profile $H_PROFILE \
    --id 1.0 \
    --input_prefix /data/input/ \
    --output_prefix /data/output/ \
    --data_dir data_hv_${profile}/ \
    --train_dir train/Annotations \
    --valid_dir test/Annotations \
    --extract \
    --data_modes train,test \
    --input_augs p_linear_2 \
    --inf_batch_size 16 \
    --inf_data_list data_hv_${profile}/test/Images/ \
    --inf_auto_metric 'valid_dice' \
    --inf_auto_find_chkpt

# Other choices:
# --inf_model dm_hv_class_${profile}.npz \

# --inf_auto_find_chkpt \
# --inf_auto_metric 'valid_dice' \

# --inf_model /data/output/export/serving/serving/variables/variables.data-00000-of-00001 \

# --preproc \
# --norm_brightness \
# --mode train \
# --image train_1 \

# --skip_types 'Misc,Spindle' \
# --outline 'Inflammatory' \
# --extract_type mirror \
# --input_norm \
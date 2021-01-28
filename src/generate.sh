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
    --input_augs p_linear_2 \
    --data_modes train,test \
    --inf_data_list data_hv_${profile}/test/Images/ \
    --inf_auto_find_chkpt \
    --inf_auto_metric 'valid_dice'

# Other choices:
# --inf_model dm_hv_class_${profile}.npz \

# --preproc \
# --norm_brightness \
# --mode train \
# --image train_1 \

# --skip_types 'Misc,Spindle' \
# --outline 'Inflammatory' \
# --extract_type mirror \
# --input_norm \
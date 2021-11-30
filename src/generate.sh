# Model type that will train or run inference
H_PROFILE=hv_consep python create_config.py \
--profile $H_PROFILE \
--id 1.0 \
--input_prefix /data/input/ \
--output_prefix /data/output/ \
--data_dir data_${H_PROFILE}/ \
--train_dir train/Annotations \
--valid_dir test/Annotations \
--extract \
--data_modes train,test \
--input_augs p_linear_2 \
--inf_batch_size 16 \
--inf_data_list data_${H_PROFILE}/test/Images/ \
--inf_auto_metric 'valid_dice' \
--inf_auto_find_chkpt


# Other choices:
# --inf_model dm_class_${H_PROFILE}.npz \

# --inf_auto_find_chkpt \
# --inf_auto_metric 'valid_dice' \

# --inf_model /data/output/export/serving/serving/variables/variables.data-00000-of-00001 \

# --skip_types 'Misc,Spindle' \
# --outline 'Inflammatory' \
# --extract_type mirror \
# --input_norm \
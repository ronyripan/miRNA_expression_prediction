# miRNA_expression_prediction
Studying miRNA activity at the single-cell level presents a significant challenge due to the limitations of existing single-cell technologies in capturing miRNAs. To address this, we introduce two deep learning models: Cross-modality (CM) and single-modality (SM), both based on encoder-decoder architectures.

Here in the "code" folder, we have four subfolders:

"bulk_sample_data" = It has two sample datasets. One is sample miRNA data; another is sample mRNA data.

"single_cell_experiment_data" = It has four sample datasets. Two sample bulk miRNA and mRNA data; two sample single-cell miRNA and mRNA data.

"CM" = It has all the codes to train the CM model.

"SM" = It has all the codes to train the SM model.



Train and test with bulk data:

Both "CM" and "SM" subfolders have "bulk_pipeline" jupyter notebooks. To train and test, just run this jupyter notebook with the data that is provided in the "bulk_sample_data" subfolder. 

Train and test with single-cell data:

Both "CM" and "SM" subfolders have "single_cell_pipeline" jupyter notebooks. To train and test, just run this jupyter notebook with the data that is provided in the "single_cell_experiment_data" subfolder. 

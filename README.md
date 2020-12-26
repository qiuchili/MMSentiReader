
# Read and Process Multimodal Sentiment Analysis and Emotion Recognition Data

Contributor: Qiuchi Li

## Instructions to run the code

### For a single function
1. Set up the configurations in config/config_of_the_run.ini
2. python run.py -config config/config_of_the_run.ini

### For processing CMU-Multimodal SDK data
1. python run.py -config config/download.ini (Data download)
2. python run.py -config config/align.ini (Data alignment)
3. python run.py -config config/extract_cmusdk.ini (Extract data to pickle file)

### For processing Conv-Emotion data
1. Download data from **https://github.com/declare-lab/conv-emotion**
3. python run.py -config config/pretrain_visual.ini (Optional, pretrain visual features, only for MELD)
4. python run.py -config config/add_act.ini (Optional, adding dialogue act annotations)
5. python run.py -config config/extract_conv_emotion.ini (Extract data to pickle file)

## Configuration setup

#### Adding dialogue act to the dataset (For dialogue datasets)
  + python run.py -config config/add_act.ini
  + **mode = add_act**.
  + **dataset_name in `{'meld','iemocap'}`**. Name of the dataset.
  + **in_path**. Pickle file storing the original data (with no act information).
  + **out_path**. Path storing the output pickle data (with act information).
  + **act_annotation_path**. CSV file storing the dialogue act annotations.

#### Aligning multimodel sequence (For monologue datasets, CMU-MultimodalSDK Utility)
  + python run.py -config config/align.ini
  + **mode = align**.
  + **dataset_name in `{'cmumosei','cmumosi','iemocap'}`**. Name of the dataset.
  + **dataset_type = multimodal**.
  + **datasets_dir**. Path storing all the datasets.
  + **align_function = avg**. Alignment function. Only average is supported for the moment.
  + **modality in `{'acoustic', 'visual', 'textual'}`**. The pivot modality for alignment. Other modalities will be aligned to this modality.
  + **align_output_dir**. Output directory storing the aligned features. Must be an existing path.

#### Download CMU-MultimodalSDK datasets (CMU-MultimodalSDK Utility)
  + python run.py -config config/download.ini
  + **mode = download**.
  + **dataset_name in `{'cmumosei','cmumosi','iemocap'}`**. Name of the dataset.
  + **datasets_dir**. Path where the dataset will be stored.

#### Extracting CMU-MultimodalSDK dataset to pickle file
  + python run.py -config config/extract_cmusdk.ini
  + **mode = extract_cmusdk**.
  + **datasets_dir**. Path storing all the datasets.
  + **data_aligned in `{'True', 'False'}`**. Whether the data is in an aligned format.
  + **data_dir**. Name of the folder storing the dataset in CMU-MultimodalSDK format.
  + **dataset_type = multimodal**.
  + **dataset_name in `{'cmumosei','cmumosi','iemocap'}`**. Name of the dataset.
  + **label = sentiment**.
  + **max_seq_len**. Length of a multimodal sequence. Zero-padding shorter sequence and truncating longer sequence.
  + The output pickle file will be **self.datasets_dir/cmusdk_data/`{'self.dataset_name'}`_`{'self.label'}`.pkl**

#### Extracting dialogue datasets in **https://github.com/declare-lab/conv-emotion** to pickle file
  + python run.py -config config/extract_conv_emotion.ini
  + **mode = extract_conv_emotion**.
  + **in_path**. Pickle file storing the original data in Conv-Emotion format.
  + **out_path**. Path storing the output pickle data.
  + **dataset_name in `{'meld','iemocap'}`**. Name of the dataset.
  + **max_seq_len**. Length of a multimodal sequence. Zero-padding shorter sequence and truncating longer sequence.

#### Pretraining Visual Features for MELD
  + python run.py -config config/pretrain_visual.ini
  + **mode = extract_raw_feature**.
  + **file_path**. Pickle file storing the original data in Conv-Emotion format.
  + **stored_path**. Folder of pickle files storing video clip data.
  + **videos_dir**. Folder storing raw MELD video clips.
  + **output_path**. Path storing the output data with pre-trained visual features, in Conv-Emotion format.
  + Other network configurations.



  




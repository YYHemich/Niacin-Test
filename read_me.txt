Download the dataset and put it into the folder "datasets" (Just put the negative sample and the positive folder into the folder "datasets")

train_CNN.py is to train the CNN features extractor.
feature_extract.py is to extract the features by a trained CNN extractor.
train_LSTM is to train the LSTM with the extracted features.

in_dict.pkl and sort_dict.pkl are two files that align the images time points of each subject.
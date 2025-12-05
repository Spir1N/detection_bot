echo "Downloading a dataset..."

mkdir datasets

curl -L -o pascal-voc-2012-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset
unzip pascal-voc-2012-dataset.zip -qd datasets/pascal-voc-2012-dataset
rm -r pascal-voc-2012-dataset.zip

echo "The download is complete!"
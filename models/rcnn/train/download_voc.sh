echo "Downloading a dataset..."

mkdir datasets

wget https://ultralytics.com/assets/VOCtrainval_06-Nov-2007.zip
unzip -qd  datasets VOCtrainval_06-Nov-2007.zip
rm -r VOCtrainval_06-Nov-2007.zip

wget https://ultralytics.com/assets/VOCtest_06-Nov-2007.zip
unzip -qd  datasets VOCtest_06-Nov-2007.zip
rm -r VOCtest_06-Nov-2007.zip

wget https://ultralytics.com/assets/VOCtrainval_11-May-2012.zip
unzip -qd  datasets VOCtrainval_11-May-2012.zip
rm -r VOCtrainval_11-May-2012.zip

echo "The download is complete!"
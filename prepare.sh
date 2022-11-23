pip install pytorch-crf
cp ../input/seqlabel/* ./
mkdir data
mv *.txt data/
python main.py
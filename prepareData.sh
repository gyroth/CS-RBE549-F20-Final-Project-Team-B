# !/bin/bash

rm -r dice;
cp -r ../CS-RBE549-F20-Final-Project-Team-B/dice/train dice;
rm -r train2017
mkdir -p train2017/images
mkdir train2017/labels
find dice/ -name '*.jpg' -exec mv {} train2017/images \;
find dice/ -name '*.txt' -exec mv {} train2017/labels \;

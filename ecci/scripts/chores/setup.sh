#!/bin/bash

#echo "Creating a new conda env named 'eCCI'..."
#conda create --name eCCI python=3.7
export ecci_pip=$HOME/.user_conda/miniconda/envs/eCCI/bin/pip

echo "Using pip located at $ecci_pip ?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

echo "Installing EDS-Toolbox..."
$ecci_pip install edstoolbox

echo "Creating a Jupyter Kernel for the eCCI env..."
eds-toolbox kernel --spark --hdfs

cd
cd ecci
echo "Installing the requirements..."
$ecci_pip install -r requirements.txt

echo "Installing the library..."
$ecci_pip install -e .

git config --local credential.helper ''
git config user.email "thomas.petitjean@aphp.fr"
git config user.name "Thomas PETIT-JEAN"

#!/bin/bash

export ecci_pip=$HOME/.user_conda/miniconda/envs/eCCI/bin/pip

case $1 in

  edsnlp)
    $ecci_pip install --force-reinstall --no-deps git+https://gitlab.eds.aphp.fr/equipedatascience/ecci_edsnlp.git@comorb_last_version
    ;;

  labeltool)
    $ecci_pip install --force-reinstall --no-deps labeltool
    ;;

  all | "")
    $ecci_pip install --force-reinstall --no-deps git+https://gitlab.eds.aphp.fr/equipedatascience/ecci_edsnlp.git@comorb_last_version
    $ecci_pip install --force-reinstall --no-deps labeltool
    ;;

  *)
    echo -n "Unknown"
    ;;
esac

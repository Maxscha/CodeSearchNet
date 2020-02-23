python predict.py -r $1 -l python --folder_save --no_upload
python predict.py -r $1 -l go --folder_save --no_upload
python predict.py -r $1 -l javascript --folder_save --no_upload
python predict.py -r $1 -l java --folder_save --no_upload
python predict.py -r $1 -l php --folder_save --no_upload
python predict.py -r $1 -l ruby --folder_save --no_upload

python submit_predict.py -r $1
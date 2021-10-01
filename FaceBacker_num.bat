@REM %1 name 
@REM %2 mode [train | test]
@REM @REM %3 epochs
python main.py -i "./train_dataset/KNU_face_dataset" -n %1 -m %2 -e 10 --use_numeric


@REM COMA Version
@REM python main.py  -n %1 -m %2 -e 10 --use_numeric
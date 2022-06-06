# Bat script to launch each model in an ensemble through FEFLOW

@echo off

set /A assim_number=0
set /A max_numero_of_model=155

call E:\Anaconda/Scripts/activate.bat E:\Anaconda

if exist "D:\Dropbox\FEFLOW_MDA_MPMW_500_20_amplifie\Data\Assim_%assim_number%\Individual_Update\" (
    echo Individual_Update folder exists in Data\Assim_%assim_number%
) else (
    echo Error, Individual_Update folder is missing in Data\Assim_%assim_number%
    pause
    exit 
 
)

if exist "D:\Dropbox\FEFLOW_MDAFEFLOW_MDA_MPMW_500_20_amplifie_MPMW\Data\Assim_%assim_number%\Head\" (
    echo Head folder exists in Data\Assim_%assim_number%
) else (
    echo Error, Head folder is missing in Data\Assim_%assim_number%
    pause
    exit 
 
)

for /l %%x in (0, 1, %max_numero_of_model%) do (

    if exist "D:\Dropbox\FEFLOW_MDA_MPMW_500_20_amplifie\Data\Assim_%assim_number%\Head\head_%%x.npy" (
    echo Loading succesfull, permeability field %%x already run ) else (

        if exist "D:\Dropbox\FEFLOW_MDA_MPMW_500_20_amplifie\Data\Assim_%assim_number%\Individual_Update\assim_%assim_number%_model_%%x.npy" (
        echo Loading succesfull, permeability field %%x being run 
        python D:\Dropbox\FEFLOW_MDA_MPMW_500_20_amplifie\Functions\Launch_Feflow.py %assim_number% %%x ) else (
        echo Loading unsuccesfull, permeability field assim_%assim_number%_model_%%x.npy is missing. Could be normal in case of dropped non converging model
        )
    )
)

pause
@echo off
REM ======================================================================
REM  Fruit & Vegetable Freshness Detection System - Windows helper script
REM
REM    run.bat setup        install Python 3.12 (if missing), create .venv, install packages, check GPU
REM    run.bat download     download the Kaggle dataset into dataset\raw
REM    run.bat preprocess   clean, resize, de-duplicate and split the dataset
REM    run.bat train        train the model (GPU if available)
REM    run.bat evaluate     evaluate on the test split, write results\
REM    run.bat predict IMG  predict one image, e.g. run.bat predict sample_images\xyz.jpg
REM    run.bat test         automatic web-app test with the sample images
REM    run.bat app          start the web app on http://127.0.0.1:5000
REM    run.bat all          setup + download + preprocess + train + evaluate + test
REM
REM  Every step also saves its console output in logs\<step>.log
REM ======================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist logs mkdir logs

set "STEP=%~1"
if "%STEP%"=="" goto menu
if /i "%STEP%"=="setup"      goto do_setup
if /i "%STEP%"=="download"   goto do_download
if /i "%STEP%"=="preprocess" goto do_preprocess
if /i "%STEP%"=="train"      goto do_train
if /i "%STEP%"=="evaluate"   goto do_evaluate
if /i "%STEP%"=="predict"    goto do_predict
if /i "%STEP%"=="test"       goto do_test
if /i "%STEP%"=="app"        goto do_app
if /i "%STEP%"=="all"        goto do_all
echo Unknown step "%STEP%". Open run.bat to see the list of steps.
exit /b 1

:menu
echo.
echo  Fruit ^& Vegetable Freshness Detection System
echo  --------------------------------------------
echo   1  setup      (install Python packages, check GPU)
echo   2  download   (Kaggle dataset, ~5.2 GB)
echo   3  preprocess
echo   4  train
echo   5  evaluate
echo   6  test       (web app self-test)
echo   7  app        (start web app)
echo   8  all        (1 to 6 in one go)
echo.
set /p "CHOICE=Choose a number: "
if "%CHOICE%"=="1" goto do_setup
if "%CHOICE%"=="2" goto do_download
if "%CHOICE%"=="3" goto do_preprocess
if "%CHOICE%"=="4" goto do_train
if "%CHOICE%"=="5" goto do_evaluate
if "%CHOICE%"=="6" goto do_test
if "%CHOICE%"=="7" goto do_app
if "%CHOICE%"=="8" goto do_all
echo Invalid choice.
pause
exit /b 1

REM ----------------------------------------------------------------------
:do_setup
call :setup
goto finish

:do_download
call :run download src\download_dataset.py
goto finish

:do_preprocess
call :run preprocess src\data_preprocessing.py
goto finish

:do_train
call :run train src\train.py
goto finish

:do_evaluate
call :run evaluate src\evaluate.py
goto finish

:do_test
call :run test_web_app tests\test_web_app.py
goto finish

:do_predict
if "%~2"=="" (
  echo Usage: run.bat predict path\to\image.jpg
  exit /b 1
)
call :need_venv || exit /b 1
"%VENV_PY%" src\predict.py "%~2"
exit /b %errorlevel%

:do_app
call :need_venv || exit /b 1
"%VENV_PY%" app\app.py
exit /b %errorlevel%

:do_all
echo %date% %time% pipeline started> logs\pipeline_status.txt
call :setup                                   || goto failed
call :run download src\download_dataset.py    || goto failed
call :run preprocess src\data_preprocessing.py || goto failed
call :run train src\train.py                  || goto failed
call :run evaluate src\evaluate.py            || goto failed
call :run test_web_app tests\test_web_app.py  || goto failed
echo %date% %time% ALL STEPS FINISHED>> logs\pipeline_status.txt
echo.
echo ==== All steps finished. Start the web app with:  run.bat app ====
goto finish

:failed
echo %date% %time% PIPELINE STOPPED - see the last log in logs\>> logs\pipeline_status.txt
echo.
echo ==== A step failed. Scroll up or open the newest file in logs\ ====
goto finish

:finish
set "RC=%errorlevel%"
if "%STEP%"=="" pause
exit /b %RC%

REM ======================================================================
REM  Subroutines
REM ======================================================================

REM  :run <log name> <python script> -> runs it with the venv Python, output also saved to logs\<name>.log
:run
call :need_venv || exit /b 1
echo.
echo ==== %~1 ====
echo %date% %time% %~1 started>> logs\pipeline_status.txt
"%VENV_PY%" tools\run_logged.py "logs\%~1.log" "%VENV_PY%" %~2
if errorlevel 1 (
  echo %date% %time% %~1 FAILED>> logs\pipeline_status.txt
  exit /b 1
)
echo %date% %time% %~1 OK>> logs\pipeline_status.txt
exit /b 0

:need_venv
if exist "%VENV_PY%" exit /b 0
echo The virtual environment does not exist yet. Run:  run.bat setup
exit /b 1

:setup
echo.
echo ==== setup ====
echo %date% %time% setup started>> logs\pipeline_status.txt
for /f "delims=" %%A in ('powershell -NoProfile -Command "[math]::Round((Get-Item .).PSDrive.Free/1GB,1)"') do set "FREE_GB=%%A"
echo Free disk space on this drive: %FREE_GB% GB (about 20 GB is needed in total)
echo Free disk space: %FREE_GB% GB> logs\setup_system.log
where nvidia-smi >nul 2>nul && nvidia-smi >> logs\setup_system.log 2>&1

if exist "%VENV_PY%" goto setup_packages
call :find_python || exit /b 1
echo Creating virtual environment .venv with "%BASE_PY%"
"%BASE_PY%" -m venv .venv || exit /b 1

:setup_packages
echo Installing packages - PyTorch with CUDA is about 3 GB, this can take 10-30 minutes...
"%VENV_PY%" -m pip install --upgrade pip >> logs\setup_system.log 2>&1
"%VENV_PY%" tools\run_logged.py logs\setup_packages.log "%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo %date% %time% setup FAILED>> logs\pipeline_status.txt
  exit /b 1
)
"%VENV_PY%" tools\run_logged.py logs\check_gpu.log "%VENV_PY%" src\check_gpu.py
if errorlevel 1 (
  echo %date% %time% setup FAILED>> logs\pipeline_status.txt
  exit /b 1
)
echo %date% %time% setup OK>> logs\pipeline_status.txt
exit /b 0

REM  :find_python -> sets BASE_PY to a Python 3.12 interpreter, installing it with winget if needed
:find_python
set "BASE_PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "BASE_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined BASE_PY exit /b 0
where py >nul 2>nul && for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "BASE_PY=%%P"
if defined BASE_PY exit /b 0
echo Python 3.12 was not found. Installing it with winget (from python.org, current user only)...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements >> logs\setup_system.log 2>&1
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "BASE_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined BASE_PY exit /b 0
echo Could not install Python automatically. Install Python 3.12 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then run:  run.bat setup
exit /b 1

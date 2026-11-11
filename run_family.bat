@echo off
REM ConvNeXt family experiments under the PV protocol.
REM Run from the experiments folder. Stops at the first failed run.
REM Expect tens of minutes per run on an RTX 3060; convnext_base is the slowest.

set DATA=E:\0_paper15_pharamaceutical_tablets\experiments\datasets\pharmacy\pv_augmented

REM 1. The family, each with the CE baseline head and the proposed EDL head.
for %%B in (convnext_atto convnext_femto convnext_tiny convnext_small convnext_base) do (
  for %%H in (ce edl) do (
    echo.
    echo === %%B  head=%%H ===
    python train.py --train-root "%DATA%" --backbone %%B --head %%H --num-workers 0
    if errorlevel 1 goto :fail
  )
)

REM 2. Variants of ConvNeXt-Tiny for the design step.
echo.
echo === convnext_tiny_22k  head=edl (ImageNet-22k pretraining) ===
python train.py --train-root "%DATA%" --backbone convnext_tiny_22k --head edl --num-workers 0
if errorlevel 1 goto :fail

echo.
echo === convnext_tiny  head=edl  strides 4 8 16 32 (hairline-crack tap) ===
python train.py --train-root "%DATA%" --backbone convnext_tiny --head edl --strides 4 8 16 32 --tag s4 --num-workers 0
if errorlevel 1 goto :fail

echo.
python summarize.py
echo FAMILY RUNS FINISHED
goto :eof

:fail
echo.
echo A RUN FAILED. Copy the output above and paste it into the chat.
exit /b 1

@echo off
REM Three-seed run of the proposed model under the PV protocol.
REM Seed 7 is the protocol's own seed. Seeds 42 and 123 are additional seeds
REM added at reviewer request, and they must be reported as such.
REM If seed 7 has already been trained, change the list below to: 42 123

set DATA=E:\0_paper15_pharamaceutical_tablets\experiments\datasets\pharmacy\pv_augmented

for %%S in (7 42 123) do (
  echo.
  echo === proposed model  convnext_tiny  edl  amsm_only  seed=%%S ===
  python train.py --train-root "%DATA%" --backbone convnext_tiny --head edl --variant amsm_only --seed %%S --num-workers 0
  if errorlevel 1 goto :fail
)

echo.
python seed_summary.py --runs runs
echo SEED RUNS FINISHED
goto :eof

:fail
echo.
echo A RUN FAILED. Copy the output above and paste it into the chat.
exit /b 1

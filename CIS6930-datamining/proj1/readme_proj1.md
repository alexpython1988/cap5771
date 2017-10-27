**How to run in linux system with default R and Rscript installed:**
1. Open a terminal in current directory
```sh
ll
```
2. You should see two .R files as proj1_s.R and proj1_m.R. The *_s.R is the project written with single thread and *_m.R is the project utilized the multi-process. The multi-process script requires to install devstool package which might cause problem in linux env, so I suggest to run the single thread script first unless it took too long time.
note: before run script, make sure the code setwd() is commented 
```sh
sudo Rscript proj1_s.R    
```
3. Wait until the results are printed in the console

**How to run in RStudio:**
1. Open file proj1_s.R in the RStudio
2. Select all lines of code then click run
3. Wait until the results are printed in the console

_note_: 
> due to the cross validation and tunelength, the program will run for a while and may take as long as 10 minutes to finish depending on cpu speed. There are some warning will be printed before the final results, please ignore all those logs.
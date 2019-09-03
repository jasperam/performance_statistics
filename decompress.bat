rem author: Jacob Bishop

set rootDir=%1
set sevenZ=%2
set workDir=%3
set targetFile=%4
set saveDir=%5

rem echo %rootDir%
rem echo %sevenZ%
rem echo %workDir%
rem echo %targetFile%
rem echo %saveDir%

%rootDir%:
cd %workDir%
echo %sevenZ% x %targetFile% -o%saveDir%
%sevenZ% x %targetFile% -o%saveDir%


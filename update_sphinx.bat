@echo off
cd docs/sphinx
call make html
robocopy build\html ..\sphinx /E /MOVE
echo Documentation updated!